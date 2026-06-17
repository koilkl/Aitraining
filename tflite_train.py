import tensorflow as tf
import numpy as np
import os
from tensorflow.lite.python.util import convert_bytes_to_c_source


# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --------------------------
# 1) Data preparation (MNIST example)
# --------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# --------------------------
# 2) Build a lightweight CNN (ESP-friendly)
# --------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1), name="input"),
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax", name="output")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=8,
    validation_split=0.1,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nOriginal model test accuracy: {test_acc:.4f}")

# --------------------------
# 3) Convert to int8 fully-integer quantized TFLite model
# --------------------------
print("\n=== Converting to int8 fully-integer quantized model ===")

# Representative dataset generator (required for full-integer quantization calibration)
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(200):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Important: use integer-only ops compatible with TFLite Micro
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save TFLite file
with open("g_model.tflite", "wb") as f:
    f.write(tflite_model)

model_size = os.path.getsize("g_model.tflite") / 1024
print(f"Model size: {model_size:.2f} KB")
print("Saved as: g_model.tflite")

# Continue

# Export to C++ source and header
source_code, header_code = convert_bytes_to_c_source(
    tflite_model,
    array_name="g_model",
    include_guard="MODEL_H",
    use_tensorflow_license=False
)

# Write model.h
with open("model.h", "w") as f:
    f.write(header_code)

# Write model.cpp
with open("model.cpp", "w") as f:
    f.write('#include "model.h"\n\n')
    f.write(source_code)

print("\n=== Export complete ===")
print("Generated files:")
print("- model.h")
print("- model.cpp")
