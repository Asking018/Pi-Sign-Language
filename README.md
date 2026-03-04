# Pi Sign Language Translator

A web-based sign language translation application built using Flask, TensorFlow, and MediaPipe Hands.

## Features

- Real-time hand gesture detection using MediaPipe Hands
- Sign language translation using a trained TensorFlow model
- Support for American Sign Language (ASL) alphabet
- Integration with Pi Network SDK for authentication
- Mobile-friendly responsive design

## Technologies

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: TensorFlow, MediaPipe Hands
- **Authentication**: Pi Network SDK

## Getting Started

### Prerequisites

- Python 3.7 or newer
- Flask
- TensorFlow 2.x
- PIL (Pillow)
- Internet connection for MediaPipe libraries

### Installation

1. Clone the repository
   ```
   git clone https://github.com/Asking018/Pi-Sign-Language.git
   cd Pi-Sign-Language
   ```

2. Install required packages
   ```
   pip install -r requirements.txt
   ```

3. Run the application
   ```
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Allow camera access when prompted
2. Position your hand clearly in the camera view
3. Make ASL letter signs with your hand
4. See real-time translation of your signs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [TensorFlow](https://www.tensorflow.org/) for the ML framework
- [Pi Network](https://minepi.com/) for authentication SDK 