#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed ResNet-101 (bottleneck) training script using tf.data pipelines.

Save as train_resnet_fixed.py and run:
    python train_resnet_fixed.py
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, MaxPooling2D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import time

# -----------------------------
# Config - edit as needed
# -----------------------------
TRAIN_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
VAL_DIR   = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\val_sorted"
WEIGHTS_OUT = "Weights_Resnet101.h5"
IMG_SIZE = (128, 128)
CHANNELS = 1  # 1 = grayscale, 3 = RGB
NUM_CLASSES = 100
BATCH_SIZE = 64
MAX_EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE
USE_MIXED_PRECISION = True
AUGMENT = True

# -----------------------------
# GPU setup: memory growth
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"{len(gpus)} GPU(s) available.")
    except Exception as e:
        print("Could not set GPU memory growth:", e)
else:
    print("No GPU found — running on CPU.")

# -----------------------------
# Optional: mixed precision
# -----------------------------
if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled (policy = mixed_float16).")
    except Exception as e:
        print("Mixed precision not enabled:", e)
        USE_MIXED_PRECISION = False

# -----------------------------
# ResNet bottleneck block (correct)
# -----------------------------
def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """
    Bottleneck block:
      - 1x1 reduce -> filters
      - 3x3 conv  -> filters
      - 1x1 expand -> filters * 4 (out_channels)
    Shortcut is adjusted to match out_channels when needed.
    """
    shortcut = x
    out_filters = filters * 4

    # 1x1 reduce
    x = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 conv
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 expand
    x = Conv2D(out_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # adjust shortcut if needed
    if downsample or int(shortcut.shape[-1]) != out_filters:
        shortcut = Conv2D(out_filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# -----------------------------
# ResNet-101 builder (fixed)
# -----------------------------
def build_resnet101(input_shape=(128,128,1), num_classes=100, dropout_rate=0.3):
    """
    Build ResNet-101 (bottleneck) with layer config [3,4,23,3].
    filters for stages: 64, 128, 256, 512 (these are the 'filters' passed to bottleneck;
    bottleneck expands them by 4 internally).
    """
    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # stage_configs: (filters, blocks, first_stage_downsample_stride)
    stage_configs = [
        (64, 3),    # stage 1
        (128, 4),   # stage 2
        (256, 23),  # stage 3 (heavy)
        (512, 3),   # stage 4
    ]

    for stage_idx, (filters, blocks) in enumerate(stage_configs):
        # first block in each stage: downsample for stage_idx != 0
        first_stride = 1 if stage_idx == 0 else 2
        # first block with downsample=True when not the first stage
        x = bottleneck_block(x, filters, stride=first_stride, downsample=(stage_idx != 0))
        for _ in range(blocks - 1):
            x = bottleneck_block(x, filters, stride=1, downsample=False)

    x = GlobalAveragePooling2D()(x)
    if dropout_rate and dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    dense_dtype = 'float32' if USE_MIXED_PRECISION else None
    outputs = Dense(num_classes, activation='softmax', dtype=dense_dtype)(x)

    model = Model(inputs, outputs, name='ResNet101_bottleneck_fixed')
    return model

# -----------------------------
# Data pipeline utilities
# -----------------------------
def prepare_datasets(train_dir, val_dir, img_size=(128,128), batch_size=32, channels=1, augment=False):
    """
    Create train/val tf.data.Dataset using image_dataset_from_directory.
    Returns: train_ds, val_ds
    """
    color_mode = 'grayscale' if channels == 1 else 'rgb'

    if not os.path.isdir(train_dir):
        raise ValueError(f"train_dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise ValueError(f"val_dir not found: {val_dir}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    # Normalize images to [0,1]; ensure grayscale channel if requested
    def rescale(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # image_dataset returns shape (H,W,1) for grayscale; convert if necessary
        if channels == 1 and tf.shape(image)[-1] == 3:
            image = tf.image.rgb_to_grayscale(image)
        return image, label

    train_ds = train_ds.map(rescale, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(rescale, num_parallel_calls=AUTOTUNE)

    if augment:
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.08)
            return image, label
        train_ds = train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    # Prefetch to improve performance (do not cache by default to avoid memory issues)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds

# -----------------------------
# Learning rate scheduler
# -----------------------------
def lr_schedule(epoch):
    if epoch > 0.9 * MAX_EPOCHS:
        return 1e-5
    elif epoch > 0.75 * MAX_EPOCHS:
        return 1e-4
    elif epoch > 0.5 * MAX_EPOCHS:
        return 5e-4
    else:
        return 5e-3

# -----------------------------
# Plot helper
# -----------------------------
def plot_history(history, out_prefix="resnet101"):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{out_prefix}_loss.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10,5))
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    plt.plot(epochs, history.history.get(acc_key, []), label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{out_prefix}_accuracy.png', dpi=300)
    plt.close()

# -----------------------------
# Train routine (main)
# -----------------------------
def train():
    train_ds, val_ds = prepare_datasets(
        TRAIN_DIR, VAL_DIR,
        img_size=IMG_SIZE, batch_size=BATCH_SIZE,
        channels=CHANNELS, augment=AUGMENT
    )

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    model = build_resnet101(input_shape=input_shape, num_classes=NUM_CLASSES, dropout_rate=0.3)
    model.summary()

    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_cb = TensorBoard(log_dir=os.path.join('tensorboard_log', timestamp), histogram_freq=1)
    lr_cb = LearningRateScheduler(lr_schedule)
    ckpt_cb = ModelCheckpoint(WEIGHTS_OUT, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    print("Starting training ...")
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=MAX_EPOCHS,
        validation_data=val_ds,
        callbacks=[tb_cb, lr_cb, ckpt_cb],
        verbose=1
    )

    plot_history(history, out_prefix="resnet101_fixed")
    print(f"Training complete. Best weights (if any) saved to: {WEIGHTS_OUT}")
    end = time.time()
    elapsed = end-start
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes, {elapsed/3600:.2f} hours)")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train()
