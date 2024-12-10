import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3

class InceptionClassifier:
    def __init__(self, input_shape=(299, 299, 3)):
        self.input_shape = input_shape

    def build_model(self, num_classes):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False  # Freeze base model layers

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')  # Output layer for `n` classes
        ])

        return model

    def compile_model(self, model, learning_rate=0.001):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
