import tensorflow as tf
import tf2onnx
import onnx

# Load the TensorFlow model
model = tf.keras.models.load_model('models/best_model.h5')

# Convert the model to ONNX format
spec = (tf.TensorSpec((None, 64, 64, 1), tf.float32, name="input"),)
output_path = "models/sign_language_model.onnx"

# Convert the model
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"Model has been converted and saved to {output_path}") 