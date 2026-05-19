import tensorflow as tf
from tensorflow.keras import layers

# 1. 训练模型（和原库用法一致）
model = tf.keras.Sequential([layers.Dense(1)])
model.fit(...)

# 2. 直接转换为TFLite（核心功能，替代旧库）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存模型
with open("model.tflite", "wb") as f:
    f.write(tflite_model)