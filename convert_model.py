import tensorflow as tf
from keras.models import load_model
import os

def convert_model():
    """Convert the trained model to a format compatible with TensorFlow.js"""
    # Load the trained model
    model_path = os.path.join('models', 'best_model.h5')
    model = load_model(model_path)
    
    # Create output directory if it doesn't exist
    output_dir = 'static'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model in TensorFlow SavedModel format
    tf_savedmodel_path = os.path.join(output_dir, 'model')
    model.save(tf_savedmodel_path, save_format='tf')
    print(f"Model saved to {tf_savedmodel_path}")

if __name__ == "__main__":
    convert_model() 