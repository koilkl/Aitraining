import tensorflow as tf
import numpy as np
import os
from tensorflow.lite.python.util import convert_bytes_to_c_source


# 设置随机种子保证可复现
tf.random.set_seed(42)
np.random.seed(42)

# --------------------------
# 1. 数据准备（MNIST示例）
# --------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# --------------------------
# 2. 构建轻量级CNN（专为ESP32优化）
# --------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1), name="input"),
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),  # 减少通道数
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),  # 减少全连接层大小
    tf.keras.layers.Dense(10, activation="softmax", name="output")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",  # 不用one-hot，节省内存
    metrics=["accuracy"]
)

# 训练
model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=8,
    validation_split=0.1,
    verbose=1
)

# 评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n原始模型测试准确率: {test_acc:.4f}")

# --------------------------
# 3. 转换为ESP32-P4专用的int8全整数量化模型
# --------------------------
print("\n=== 转换为int8全整数量化模型 ===")

# 代表性数据集生成器（必须用于全整数量化校准）
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(200):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# 关键：仅使用TFLite Micro支持的整数操作集
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # ESP32-P4推荐int8（比uint8更通用）
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 保存TFLite文件
with open("g_model.tflite", "wb") as f:
    f.write(tflite_model)

model_size = os.path.getsize("g_model.tflite") / 1024
print(f"模型大小: {model_size:.2f} KB")
print(f"模型已保存为: g_model.tflite")

# 接上面的代码，继续执行

# 转换为C++源文件和头文件
source_code, header_code = convert_bytes_to_c_source(
    tflite_model,
    array_name="g_model",  # 数组名称
    include_guard="MODEL_H",  # 头文件保护宏
    use_tensorflow_license=False
)

# 写入model.h
with open("model.h", "w") as f:
    f.write(header_code)

# 写入model.cpp
with open("model.cpp", "w") as f:
    f.write('#include "model.h"\n\n')
    f.write(source_code)

print("\n=== 导出完成 ===")
print("生成的文件:")
print("- model.h")
print("- model.cpp")