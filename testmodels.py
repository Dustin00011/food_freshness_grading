import tensorflow as tf

custom_model = tf.keras.models.load_model(
    "models/best_custom_cnn_fixed.keras",
    compile=False
)

mobile_model = tf.keras.models.load_model(
    "models/best_mobilenetv2.keras",
    compile=False
)

print("Custom CNN:", custom_model.input_shape, custom_model.output_shape)
print("MobileNetV2:", mobile_model.input_shape, mobile_model.output_shape)

print("Both models loaded successfully!")