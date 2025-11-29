"""
Minimal Model Training Script for Cattle Breed Recognition
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Selected breeds for training (only breeds with images)
SELECTED_BREEDS = ['Gir', 'Murrah', 'Red_Sindhi', 'Sahiwal', 'Tharparkar']

def create_model(num_classes=5):
    """Create simple model for small dataset"""
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, BatchNormalization
    
    inputs = Input(shape=(224, 224, 3))
    
    # Very simple architecture to memorize small dataset
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((4, 4))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((4, 4))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model

def create_sample_data():
    """Create sample training data structure"""
    for split in ['train', 'validation']:
        for breed in SELECTED_BREEDS:
            breed_dir = f'data/{split}/{breed}'
            os.makedirs(breed_dir, exist_ok=True)
            
            # Create placeholder file
            with open(f'{breed_dir}/README.txt', 'w') as f:
                f.write(f'Place {breed} images here for {split}')

def train_model():
    """Train the model with minimal configuration"""
    print("Creating model...")
    model = create_model(len(SELECTED_BREEDS))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data generators - minimal augmentation for small dataset
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        validation_split=0.15
    )
    
    # Check if data exists
    if not os.path.exists('data/train') or len(os.listdir('data/train')) == 0:
        print("No training data found. Creating sample structure...")
        create_sample_data()
        print("Please add images to data/train/{breed_name}/ folders")
        return
    
    try:
        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(224, 224),
            batch_size=2,
            class_mode='categorical',
            classes=SELECTED_BREEDS,
            shuffle=True,
            subset='training'
        )
        
        val_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(224, 224),
            batch_size=2,
            class_mode='categorical',
            classes=SELECTED_BREEDS,
            shuffle=False,
            subset='validation'
        )
        
        print("Training model...")
        
        # Get class indices from generator
        class_indices = train_generator.class_indices
        print(f"Class indices: {class_indices}")
        
        # Calculate class weights for balanced training
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        unique_classes = np.unique(train_generator.classes)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=train_generator.classes
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(unique_classes))}
        print(f"Class weights: {class_weight_dict}")
        
        history = model.fit(
            train_generator,
            epochs=100,
            validation_data=val_generator,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save model
        model.save('models/cattle_breed_model.h5')
        print("Model saved to models/cattle_breed_model.h5")
        
        # Save breed mapping (index to breed name)
        breed_mapping = {v: k for k, v in class_indices.items()}
        import json
        with open('models/breed_mapping.json', 'w') as f:
            json.dump(breed_mapping, f, indent=2)
        
        print(f"Breed mapping saved: {breed_mapping}")
        print("Training completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure you have images in the data/train folders")

if __name__ == "__main__":
    train_model()