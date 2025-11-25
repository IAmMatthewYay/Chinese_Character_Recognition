#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Dense, GlobalAveragePooling2D, Input, Add, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------------
# GPU safety setup
# ----------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) available.")
    except Exception as e:
        print("Could not set GPU memory growth:", e)
else:
    print("No GPU found — running on CPU.")

# ----------------------------------------------------------------------
# Basic ResNet block
# ----------------------------------------------------------------------
def res_block(x, filters, stride=1):
    shortcut = x

    # First conv layer
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv layer
    x = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if shape mismatch
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add and ReLU
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# ----------------------------------------------------------------------
# ResNet model (ResNet-18)
# ----------------------------------------------------------------------
def ResNet(input_shape=(128, 128, 1), num_classes=100):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Residual blocks
    x = res_block(x, 64)
    x = res_block(x, 64)

    x = res_block(x, 128, stride=2)
    x = res_block(x, 128)

    x = res_block(x, 256, stride=2)
    x = res_block(x, 256)

    x = res_block(x, 512, stride=2)
    x = res_block(x, 512)

    # Global average pooling & classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="ResNet18_Custom")

# ----------------------------------------------------------------------
# Learning-rate scheduler
# ----------------------------------------------------------------------
def lrschedule(epoch, lr):
    if epoch > 0.9 * MAX_EPOCHS:
        return 1e-5
    elif epoch > 0.75 * MAX_EPOCHS:
        return 1e-4
    elif epoch > 0.5 * MAX_EPOCHS:
        return 5e-4
    else:
        return 5e-3

# ----------------------------------------------------------------------
# Main training routine
# ----------------------------------------------------------------------
def train_model():
    train_path = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
    val_path = r"D:\Stuff\\FRC\Chinese character recognition\Deep_Learning-master\train_data\val_sorted"
    weight_path = "Weights_Resnet.h5"

    NUM_CLASSES = 100
    BATCH_SIZE = 128

    global MAX_EPOCHS
    MAX_EPOCHS = 40

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.4
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    # Build model
    model = ResNet(input_shape=(128, 128, 1), num_classes=NUM_CLASSES)
    model.summary()

    # Compile
    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    lr_cb = LearningRateScheduler(lrschedule)
    ckpt_cb = ModelCheckpoint(weight_path, monitor='val_accuracy', save_best_only=True, mode='max')
    tb_cb = TensorBoard(log_dir='./tensorboard_log/', histogram_freq=1)
    start = time.time()
    print(f"Starting training at: {start}")

    history = model.fit(
        train_gen,
        epochs=MAX_EPOCHS,
        validation_data=val_gen,
        callbacks=[lr_cb, ckpt_cb, tb_cb]
    )

    plot_training_curves(history)
    end = time.time()
    print(f"Training complete. Best weights saved to: {weight_path}")
    elapsed = end - start
    print(f"Training complete. Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes, {elapsed/3600:.2f} hours)")

# ----------------------------------------------------------------------
# Plot training curves
# ----------------------------------------------------------------------
def plot_training_curves(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history['loss'], 'g-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'orange', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.show()

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_model()