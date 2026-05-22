from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.lite.python.util import convert_bytes_to_c_source


@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 96
    color_mode: str = "rgb"
    batch_size: int = 16
    epochs: int = 10
    validation_split: float = 0.2
    seed: int = 42
    optimizer: str = "adam"
    learning_rate: float = 0.001
    conv1_filters: int = 8
    conv2_filters: int = 16
    dense_units: int = 32
    representative_samples: int = 200


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


def load_datasets(
    dataset_dir: Path, cfg: TrainConfig
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], Tuple[int, int, int]]:
    dataset_dir = dataset_dir.resolve()
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(dataset_dir),
        labels="inferred",
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
        label_mode="int",
        color_mode=cfg.color_mode,
        batch_size=cfg.batch_size,
        image_size=(cfg.img_size, cfg.img_size),
        shuffle=True,
        seed=cfg.seed,
        validation_split=cfg.validation_split,
        subset="validation",
    )
    class_names = list(train_ds.class_names)
    input_shape = (cfg.img_size, cfg.img_size, _channels(cfg.color_mode))

    def normalize(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names, input_shape


def build_model(input_shape: Tuple[int, int, int], num_classes: int, cfg: TrainConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input"),
            tf.keras.layers.Conv2D(cfg.conv1_filters, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(cfg.conv2_filters, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(cfg.dense_units, activation="relu"),
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

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
) -> TrainResult:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, labels, input_shape = load_datasets(Path(dataset_dir), cfg)
    model = build_model(input_shape, len(labels), cfg)

    model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, verbose=1)
    loss, acc = model.evaluate(val_ds, verbose=0)

    keras_model_path = run_dir / f"{model_base_name}.keras"
    model.save(str(keras_model_path))

    tflite_bytes = convert_to_int8_tflite(model, train_ds, cfg)
    tflite_path = run_dir / f"{model_base_name}.tflite"
    tflite_path.write_bytes(tflite_bytes)

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

