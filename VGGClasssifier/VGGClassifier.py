import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

class VGGClassifier:
    def __init__(self, input_shape=(224, 224, 3)):  # VGG16 uses input size 224x224
        self.input_shape = input_shape

    def build_model(self, num_classes):
        # Load VGG16 as the base model with ImageNet weights
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False  # Freeze base model layers

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),  # Global average pooling to reduce dimensionality
            layers.Dense(256, activation='relu'),  # Add a fully connected layer
            layers.Dropout(0.5),  # Dropout layer to prevent overfitting
            layers.Dense(num_classes, activation='softmax')  # Output layer with 'n' classes
        ])

        return model

    def compile_model(self, model, learning_rate=0.001):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
