# ASL Model Training

This directory contains scripts for training and preparing the ASL recognition model for the web application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Process

1. Download the dataset:
```bash
python download_dataset.py
```

2. Train the model:
```bash
python train_model.py
```

3. Convert the model to TensorFlow.js format:
```bash
python convert_model.py
```

## Model Architecture

The model uses a CNN architecture with:
- 3 convolutional blocks with batch normalization and dropout
- Dense layers for classification
- Data augmentation for better generalization
- Early stopping to prevent overfitting

## Dataset

The ASL Alphabet Dataset contains:
- 29 classes (A-Z, SPACE, DELETE, NOTHING)
- 200x200 pixel images
- Training and validation splits (80/20)

## Output

The training process will:
1. Save the best model as `models/best_model.h5`
2. Save the final model as `models/final_model.h5`
3. Convert the best model to TensorFlow.js format in `../models/asl_model/`

## Integration with Web App

The converted model will be automatically placed in the correct location for the web application to use. The web app will load the model using TensorFlow.js and use it for real-time sign language recognition. 