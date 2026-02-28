from PIL import Image, ImageDraw
import os

def create_avatar():
    # Create a new image with a white background
    size = 100
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a circle for the head
    margin = 10
    draw.ellipse([(margin, margin), (size-margin, size-margin)], fill='#14CFF4')
    
    # Draw a smaller circle for the body
    body_margin = size // 3
    draw.ellipse([(body_margin, size//2), (size-body_margin, size)], fill='#14CFF4')
    
    # Create the assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Save the image
    image.save('assets/user-avatar.png')
    print('Created user avatar placeholder image')

if __name__ == '__main__':
    create_avatar() 