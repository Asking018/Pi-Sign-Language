import tensorflowjs as tfjs
import os

def convert_model():
    # Paths
    model_path = os.path.join('models', 'best_model.h5')
    output_path = os.path.join('..', 'models', 'asl_model')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Converting model to TensorFlow.js format...")
    
    # Convert the model
    tfjs.converters.save_keras_model(
        model_path,
        output_path,
        quantization_dtype_map={'float16': '*'},
        weight_shard_size_bytes=1024 * 1024  # 1MB shards
    )
    
    print(f"Model converted and saved to {output_path}")

if __name__ == "__main__":
    convert_model() 