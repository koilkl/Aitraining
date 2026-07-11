from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.util import convert_bytes_to_c_source

from image_preprocess import (
    PREPROCESS_MODE_AUTO_BY_LABEL,
    normalize_class_preprocess_map,
    normalize_sample_preprocess_map,
    normalize_manual_roi,
    preprocess_for_label,
)


@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 96
    color_mode: str = "grayscale"
    batch_size: int = 32
    epochs: int = 20
    validation_split: float = 0.25
    seed: int = 42
    optimizer: str = "adam"
    learning_rate: float = 0.0016
    conv1_filters: int = 16
    conv2_filters: int = 32
    dense_units: int = 64
    representative_samples: int = 200
    preprocess_mode: str = PREPROCESS_MODE_AUTO_BY_LABEL
    manual_roi: Optional[Tuple[float, float, float, float]] = None
    class_preprocess: Optional[Dict[str, Dict[str, Any]]] = None
    sample_preprocess: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    use_preprocessed_dataset: bool = False


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    labels: List[str]
    keras_model_path: Path
    tflite_path: Path
    model_h_path: Path
    model_cpp_path: Path
    metrics: Dict[str, float]


def _channels(color_mode: str) -> int:
    return 1 if color_mode == "grayscale" else 3


def _list_image_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    allowed = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
    out: Dict[str, List[Path]] = {}
    for class_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        files = [p for p in sorted(class_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed]
        if files:
            out[class_dir.name] = files
    return out


def load_datasets(
    dataset_dir: Path, cfg: TrainConfig
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], tf.data.Dataset, List[str], Tuple[int, int, int], Dict[int, float]]:
    dataset_dir = dataset_dir.resolve()
    image_map = _list_image_files(dataset_dir)
    class_names = list(image_map.keys())
    total_files = sum(len(files) for files in image_map.values())
    if len(image_map) < 2:
        raise ValueError("Need at least 2 classes with images before training.")
    if total_files < 2:
        raise ValueError("Need at least 2 images before training.")
    can_split_validation = cfg.validation_split > 0 and total_files >= 3
    if can_split_validation:
        val_count = int(total_files * float(cfg.validation_split))
        train_count = total_files - val_count
        can_split_validation = val_count >= 1 and train_count >= 1
    if can_split_validation:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
            validation_split=cfg.validation_split,
            subset="training",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
            validation_split=cfg.validation_split,
            subset="validation",
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(dataset_dir),
            labels="inferred",
            class_names=class_names,
            label_mode="int",
            color_mode=cfg.color_mode,
            batch_size=cfg.batch_size,
            image_size=(cfg.img_size, cfg.img_size),
            shuffle=True,
            seed=cfg.seed,
        )
        val_ds = None
    class_names = list(train_ds.class_names)
    input_shape = (cfg.img_size, cfg.img_size, _channels(cfg.color_mode))
    class_name_list = [str(name) for name in class_names]
    preprocess_mode = str(cfg.preprocess_mode or PREPROCESS_MODE_AUTO_BY_LABEL)
    manual_roi = normalize_manual_roi(cfg.manual_roi)
    class_preprocess = normalize_class_preprocess_map(cfg.class_preprocess)
    sample_preprocess = normalize_sample_preprocess_map(cfg.sample_preprocess)
    _ = sample_preprocess
    use_preprocessed_dataset = bool(cfg.use_preprocessed_dataset)

    channels = _channels(cfg.color_mode)
    pad = max(2, int(cfg.img_size) // 12)
    padded_size = int(cfg.img_size) + pad * 2
    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.08, 0.04),
                width_factor=(-0.08, 0.04),
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomRotation(
                factor=0.035,
                fill_mode="reflect",
            ),
        ],
        name="train_augmenter",
    )

    def focus_tensor_batch(x, y):
        def _focus_one(img: np.ndarray, label: Any):
            label_idx = int(label)
            label_name = class_name_list[label_idx] if 0 <= label_idx < len(class_name_list) else ""
            return preprocess_for_label(
                img,
                out_size=int(cfg.img_size),
                color_mode=str(cfg.color_mode),
                label_name=label_name,
                preprocess_mode=preprocess_mode,
                manual_roi=manual_roi,
                class_preprocess=class_preprocess,
            )

        x = tf.map_fn(
            lambda elems: tf.numpy_function(_focus_one, [elems[0], elems[1]], Tout=tf.float32),
            (x, y),
            fn_output_signature=tf.TensorSpec((int(cfg.img_size), int(cfg.img_size), channels), tf.float32),
        )
        return x, y

    def augment(x, y):
        x, y = focus_tensor_batch(x, y)
        x = tf.image.resize_with_crop_or_pad(x, padded_size, padded_size)
        x = tf.image.random_crop(x, size=[tf.shape(x)[0], int(cfg.img_size), int(cfg.img_size), channels])
        x = augmenter(x, training=True)
        x = tf.image.random_brightness(x, max_delta=0.08)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        return x, y

    def normalize_only(x, y):
        x = tf.clip_by_value(tf.cast(x, tf.float32) / 255.0, 0.0, 1.0)
        return x, y

    def augment_preprocessed(x, y):
        x, y = normalize_only(x, y)
        x = augmenter(x, training=True)
        x = tf.image.random_brightness(x, max_delta=0.08)
        x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=tf.float32)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)
        return x, y

    total_files_f = float(total_files)
    num_classes_f = float(max(1, len(class_names)))
    class_weights: Dict[int, float] = {}
    for idx, name in enumerate(class_names):
        count = max(1, len(image_map.get(name, [])))
        class_weights[int(idx)] = total_files_f / (num_classes_f * float(count))

    if use_preprocessed_dataset:
        calibration_ds = train_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_preprocessed, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        if val_ds is not None:
            val_ds = val_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    else:
        calibration_ds = train_ds.map(focus_tensor_batch, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        if val_ds is not None:
            val_ds = val_ds.map(focus_tensor_batch, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, calibration_ds, class_names, input_shape, class_weights


def build_model(input_shape: Tuple[int, int, int], num_classes: int, cfg: TrainConfig) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(1e-4)
    conv3_filters = max(int(cfg.conv2_filters) * 2, 64)
    dense_units = max(int(cfg.dense_units), 64)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input"),
            tf.keras.layers.Conv2D(cfg.conv1_filters, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.10),
            tf.keras.layers.Conv2D(cfg.conv2_filters, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv2D(conv3_filters, (3, 3), activation="relu", padding="same", kernel_regularizer=reg),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    opt_name = cfg.optimizer.lower()
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif opt_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9)
    elif opt_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def _representative_data_gen(train_ds: tf.data.Dataset, cfg: TrainConfig):
    remaining = cfg.representative_samples
    for batch_x, _ in train_ds.unbatch().batch(1).take(cfg.representative_samples):
        yield [batch_x]
        remaining -= 1
        if remaining <= 0:
            break


def convert_to_int8_tflite(model: tf.keras.Model, train_ds: tf.data.Dataset, cfg: TrainConfig) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(train_ds, cfg)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def export_tflite_c_sources(tflite_model: bytes, array_name: str) -> Tuple[str, str]:
    array_name = array_name.strip() or "g_model"
    header_guard = f"{array_name.upper()}_H"
    source_code, header_code = convert_bytes_to_c_source(
        tflite_model,
        array_name=array_name,
        include_guard=header_guard,
        use_tensorflow_license=False,
    )
    return source_code, header_code


def train_and_export(
    dataset_dir: Path,
    run_dir: Path,
    cfg: TrainConfig,
    model_base_name: str = "model",
    array_name: str = "g_model",
    progress: Optional[Callable[[float, str], None]] = None,
) -> TrainResult:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, calibration_ds, labels, input_shape, class_weights = load_datasets(Path(dataset_dir), cfg)
    model = build_model(input_shape, len(labels), cfg)

    if progress is not None:
        progress(0.02, "Building datasets...")

    fit_kwargs = {"epochs": cfg.epochs, "verbose": 1}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds
    fit_kwargs["class_weight"] = class_weights
    if progress is not None:
        epochs_total = max(1, int(cfg.epochs))

        class _Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                try:
                    p = 0.05 + 0.75 * (float(epoch + 1) / float(epochs_total))
                    progress(p, f"Training epoch {int(epoch + 1)}/{epochs_total}...")
                except Exception:
                    pass

        callbacks: List[tf.keras.callbacks.Callback] = [_Progress()]
    else:
        callbacks = []

    monitor_loss = "val_loss" if val_ds is not None else "loss"
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_loss,
            patience=6,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0,
        )
    )
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=0,
        )
    )
    fit_kwargs["callbacks"] = callbacks

    model.fit(train_ds, **fit_kwargs)
    eval_ds = val_ds if val_ds is not None else train_ds
    if progress is not None:
        progress(0.85, "Evaluating...")
    loss, acc = model.evaluate(eval_ds, verbose=0)

    keras_model_path = run_dir / f"{model_base_name}.keras"
    model.save(str(keras_model_path))

    if progress is not None:
        progress(0.9, "Converting to int8 TFLite...")
    tflite_bytes = convert_to_int8_tflite(model, calibration_ds, cfg)
    tflite_path = run_dir / f"{model_base_name}.tflite"
    tflite_path.write_bytes(tflite_bytes)

    if progress is not None:
        progress(0.96, "Exporting sources...")
    source_code, header_code = export_tflite_c_sources(tflite_bytes, array_name=array_name)
    model_h_path = run_dir / "model.h"
    model_cpp_path = run_dir / "model.cpp"
    model_h_path.write_text(header_code, encoding="utf-8")
    model_cpp_path.write_text('#include "model.h"\n\n' + source_code, encoding="utf-8")

    (run_dir / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress is not None:
        progress(1.0, "Done.")

    return TrainResult(
        run_dir=run_dir,
        labels=labels,
        keras_model_path=keras_model_path,
        tflite_path=tflite_path,
        model_h_path=model_h_path,
        model_cpp_path=model_cpp_path,
        metrics={"val_loss": float(loss), "val_accuracy": float(acc)},
    )


def new_run_dir(base_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return base_dir / f"run_{ts}_{rand}"
