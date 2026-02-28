from PIL import Image, ImageDraw, ImageFont
import os

def create_letter_image(letter, output_path):
    # Create a new image with a white background
    width = 300
    height = 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use Arial font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except:
        font = ImageFont.load_default()
    
    # Get the size of the text
    text_width = draw.textlength(letter, font=font)
    text_height = 120  # Approximate height for Arial font
    
    # Calculate the position to center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw the letter
    draw.text((x, y), letter, fill='black', font=font)
    
    # Draw a border
    draw.rectangle([(0, 0), (width-1, height-1)], outline='black', width=2)
    
    # Save the image
    image.save(output_path)

def main():
    # Create the signs directory if it doesn't exist
    signs_dir = 'assets/signs'
    if not os.path.exists(signs_dir):
        os.makedirs(signs_dir)
    
    # Generate images for each letter A-Z
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        output_path = os.path.join(signs_dir, f'{letter}.png')
        create_letter_image(letter, output_path)
        print(f'Created placeholder image for letter {letter}')

if __name__ == '__main__':
    main() 