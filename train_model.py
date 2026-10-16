import os
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define image properties and directories
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / 'dataset' / 'train'
VAL_DIR = BASE_DIR / 'dataset' / 'valid'
MODEL_DIR = BASE_DIR / 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Data Preprocessing and Augmentation
print("Preparing dataset generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Randomly rotate images
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    horizontal_flip=True,    # Randomly flip images horizontally
    zoom_range=0.2           # Randomly zoom images
)

val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale validation data

if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset directories are required: {TRAIN_DIR} and {VAL_DIR}"
    )

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=list(train_generator.class_indices.keys())
)


# 2. Model Building (Transfer Learning with MobileNetV2)
print("Building the model...")
# Load the pre-trained MobileNetV2 model, excluding the top (classification) layers
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze the layers of the base model so they aren't trained from scratch
base_model.trainable = False

# Add custom top layers for our specific prediction task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Combine the base model and custom top layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 3. Callbacks for Training
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'plant_disease_model.keras'), 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# 4. Training (Uncomment and run when the dataset is populated)
print("-" * 50)
print("Model Built Successfully.")
print("To train the model, place your images in structured folders under dataset/train/ and dataset/valid/, then run this script.")
print("Example folder structure:")
print("dataset/")
print("  train/")
print("    Tomato___Healthy/")
print("      img1.jpg")
print("    Tomato___Late_blight/")
print("      img2.jpg")
print("-" * 50)

model.fit(
    train_generator,
    epochs=int(os.getenv("EPOCHS", "10")),
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping]
)

with open(MODEL_DIR / 'class_indices.json', 'w', encoding='utf-8') as f:
    json.dump(train_generator.class_indices, f, indent=2)
