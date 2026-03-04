import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os
from tensorflow.keras.models import load_model

def load_and_preprocess_image(image_path, target_size=(160, 160)):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def test_model():
    """Test the trained model with sample images"""
    # Load the trained model
    model_path = os.path.join('models', 'best_model.h5')
    model = load_model(model_path)
    
    # Define class indices (A-Z and space)
    class_indices = {i: chr(65 + i) for i in range(26)}  # A-Z
    class_indices[26] = ' '  # Space
    
    # Test directory containing sample images
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"Please create a directory named '{test_dir}' and add some test images.")
        return
    
    # Test each image in the test directory
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, filename)
            print(f"\nTesting image: {filename}")
            
            # Load and preprocess the image
            img_array = load_and_preprocess_image(image_path)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get the predicted letter
            predicted_letter = class_indices[predicted_class]
            
            print(f"Predicted letter: {predicted_letter}")
            print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    test_model() 