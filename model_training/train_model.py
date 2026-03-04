import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Define dataset path: use ASL_DATASET_PATH env var or default to project_root/asl_alphabet_test
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATASET_PATH = os.environ.get('ASL_DATASET_PATH', os.path.join(_project_root, 'asl_alphabet_test'))
MODEL_SAVE_PATH = os.path.join(_project_root, 'models')
BATCH_SIZE = 16  # Reduced batch size for smaller dataset
EPOCHS = int(os.environ.get('ASL_EPOCHS', '20'))  # Set ASL_EPOCHS=3 for quick run
IMG_SIZE = 160
LEARNING_RATE = 0.0005  # Reduced learning rate for better generalization

def create_data_generators():
    """Creates ImageDataGenerators for training and validation"""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],  # Added brightness variation
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

def create_model(num_classes):
    """Creates and compiles the ResNet50V2 model for ASL classification"""
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False  # Freeze base model initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def train_model():
    """Loads data, creates model, trains it, and fine-tunes"""
    try:
        print("Creating data generators...")
        train_generator, val_generator = create_data_generators()
        num_classes = len(train_generator.class_indices)

        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")

        # Compute class weights for balanced training
        print("Computing class weights...")
        class_labels = train_generator.classes
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        class_weights = dict(enumerate(class_weight_values))

        print("Creating and compiling model...")
        model = create_model(num_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
                                               monitor='val_accuracy', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        ]

        print("Starting training...")
        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        print("Saving final model...")
        # Save the final model
        model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.h5'))
        
        return model, history

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

def save_model(model):
    """Saves the trained model in TensorFlow format"""
    print("Saving model in TensorFlow format...")
    # Save in TensorFlow SavedModel format
    tf_savedmodel_path = os.path.join(MODEL_SAVE_PATH, 'saved_model')
    model.save(tf_savedmodel_path, save_format='tf')
    print(f"Model saved to {tf_savedmodel_path}")

if __name__ == "__main__":
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Train model
    model, history = train_model()

    # Save model in TensorFlow format
    save_model(model)

    print("Model training and saving completed!")
