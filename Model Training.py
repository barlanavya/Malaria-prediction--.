import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
train_data_dir = 'D:\\AD Projects\\Malaria Disease Detection using DL\\Malaria Disease Dataset\\Train'
test_data_dir = 'D:\\AD Projects\\Malaria Disease Detection using DL\\Malaria Disease Dataset\\Test'
batch_size = 32
image_size = (224, 224)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Data augmentation for testing set (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Set class_mode to 'categorical' for multiple classes
)

# Load and prepare testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Set class_mode to 'categorical' for multiple classes
)

# Define the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with a ReLU activation
x = Dense(256, activation='relu')(x)

# Add the final prediction layer
predictions = Dense(2, activation='softmax')(x)  # Use 2 units for 2 classes and softmax activation

# Combine the base model and the custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model using learning_rate instead of lr
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up model checkpoint to save the best weights
checkpoint = ModelCheckpoint(
    'malaria_detection_model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model with model checkpoint
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[checkpoint]
)

# Save the trained model
model.save('malaria_detection_model_final.h5')
