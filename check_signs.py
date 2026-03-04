import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Define dataset path (same as in train_model.py)
DATASET_PATH = 'asl_alphabet_test'

def show_example_signs():
    """Display examples of 'A' and 'P' signs from the dataset"""
    try:
        # Look for 'A' and 'P' in the dataset
        a_path = os.path.join(DATASET_PATH, 'A')
        p_path = os.path.join(DATASET_PATH, 'P')
        
        # Get example images
        a_examples = [os.path.join(a_path, f) for f in os.listdir(a_path) if f.endswith('.jpg') or f.endswith('.png')]
        p_examples = [os.path.join(p_path, f) for f in os.listdir(p_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Limit to first few examples
        a_examples = a_examples[:3] if a_examples else []
        p_examples = p_examples[:3] if p_examples else []
        
        # Create figure
        fig, axs = plt.subplots(2, max(len(a_examples), len(p_examples)), figsize=(15, 8))
        
        # Display A examples
        for i, img_path in enumerate(a_examples):
            img = Image.open(img_path)
            axs[0, i].imshow(np.array(img))
            axs[0, i].set_title(f"'A' Sign Example {i+1}")
            axs[0, i].axis('off')
        
        # Display P examples
        for i, img_path in enumerate(p_examples):
            img = Image.open(img_path)
            axs[1, i].imshow(np.array(img))
            axs[1, i].set_title(f"'P' Sign Example {i+1}")
            axs[1, i].axis('off')
        
        # Check if we have a debug image from our app
        debug_path = 'static/hand_debug.png'
        if os.path.exists(debug_path):
            # Add a new figure for the debug image
            plt.figure(figsize=(8, 8))
            debug_img = Image.open(debug_path)
            plt.imshow(np.array(debug_img))
            plt.title("Your Hand as Seen by Model")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        print("Images displayed. Close the window to continue.")
        
    except Exception as e:
        print(f"Error displaying example signs: {e}")
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset directory {DATASET_PATH} not found. Please check the path.")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")

def fix_confusion():
    """Create an improved hand image for the 'A' sign to demonstrate the fix"""
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Create a sample A sign hand image with clearer distinction from P
    img_size = 160
    image = Image.new('RGB', (img_size, img_size), color='black')
    draw = ImageDraw.Draw(image)
    
    # Example landmarks for "A" sign (normalized to 0-1 range)
    a_landmarks = [
        (0.5, 0.7),    # Wrist
        (0.45, 0.6),   # Thumb base
        (0.42, 0.5),   # Thumb mid
        (0.4, 0.4),    # Thumb tip
        (0.4, 0.4),    # Thumb tip
        (0.55, 0.6),   # Index base
        (0.55, 0.5),   # Index mid
        (0.55, 0.4),   # Index tip
        (0.55, 0.35),  # Index tip
        (0.6, 0.6),    # Middle base
        (0.6, 0.5),    # Middle mid
        (0.6, 0.4),    # Middle tip
        (0.6, 0.35),   # Middle tip
        (0.65, 0.6),   # Ring base
        (0.65, 0.5),   # Ring mid
        (0.65, 0.4),   # Ring tip
        (0.65, 0.35),  # Ring tip
        (0.7, 0.6),    # Pinky base
        (0.7, 0.5),    # Pinky mid
        (0.7, 0.4),    # Pinky tip
        (0.7, 0.35)    # Pinky tip
    ]
    
    # Scale landmarks to image dimensions
    scaled_landmarks = [(x * img_size, y * img_size) for x, y in a_landmarks]
    
    # Draw a filled hand polygon for a clear "A" sign
    # For "A", all fingers except thumb are together and closed
    hull_points = [
        scaled_landmarks[0],  # Wrist
        scaled_landmarks[4],  # Thumb tip
        scaled_landmarks[8],  # Index tip
        scaled_landmarks[12], # Middle tip
        scaled_landmarks[16], # Ring tip
        scaled_landmarks[20], # Pinky tip
        scaled_landmarks[17], # Pinky base
        scaled_landmarks[13], # Ring base
        scaled_landmarks[9],  # Middle base
        scaled_landmarks[5],  # Index base
    ]
    
    # Fill the hand with white
    draw.polygon(hull_points, fill='white')
    
    # Draw the thumb distinctly (critical for "A" sign)
    thumb_points = [scaled_landmarks[i] for i in [0, 1, 2, 3, 4]]
    draw.line(thumb_points, fill='white', width=8)
    
    # Apply slight blur for smoother edges
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Save improved hand image
    improved_path = 'static/improved_hand.png'
    image.save(improved_path)
    print(f"Improved hand image saved to: {improved_path}")
    
    # Display comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original debug image if available
    debug_path = 'static/hand_debug.png'
    if os.path.exists(debug_path):
        debug_img = Image.open(debug_path)
        axs[0].imshow(np.array(debug_img))
        axs[0].set_title("Current Hand (Confused as 'P')")
        axs[0].axis('off')
    
    # Improved image
    improved_img = Image.open(improved_path)
    axs[1].imshow(np.array(improved_img))
    axs[1].set_title("Improved Hand for 'A'")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Checking example signs from dataset...")
    show_example_signs()
    
    print("\nGenerating improved hand image for 'A' sign...")
    fix_confusion() 