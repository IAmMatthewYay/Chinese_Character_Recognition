#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Activation,
    Dense, Flatten, Dropout, Input, PReLU
)
from tensorflow.keras.models import Model
import time

# -------------------------
# Config
# -------------------------
TRAIN_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
VAL_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\val_sorted"
OUTPUT_WEIGHTS = "Weights_Alex_Net.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 100   # default; will be overridden if folder count differs
LR_INIT = 0.01
MOMENTUM = 0.9

# -------------------------
# AlexNet
# -------------------------
def bn_relu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def Alex_model(out_dims, input_shape=(128, 128, 1)):
    input_dim = Input(input_shape)
    x = Conv2D(96, (20, 20), strides=(2, 2), padding='valid')(input_dim)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (5, 5), padding='same')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(384, (3, 3), padding='same')(x)
    x = PReLU()(x)

    x = Conv2D(384, (3, 3), padding='same')(x)
    x = PReLU()(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)

    x = Dense(4096)(x)
    x = Dropout(0.2)(x)

    x = Dense(4096)(x)
    x = Dropout(0.25)(x)

    x = Dense(out_dims)(x)
    output = Activation('softmax')(x)

    return Model(inputs=input_dim, outputs=output, name="AlexNet_Custom_128")

# -------------------------
# Data generators
# -------------------------
print("==> Preparing data generators")

train_aug = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=8,
    width_shift_range=0.06,
    height_shift_range=0.06,
    shear_range=0.05,
    zoom_range=0.05,
    validation_split=0.15  # used only if VAL_DIR absent
)

val_aug = ImageDataGenerator(rescale=1.0/255.0)

# Choose whether a separate val folder exists
use_separate_val = os.path.isdir(VAL_DIR) and any(os.scandir(VAL_DIR))

if use_separate_val:
    print("Using separate validation folder:", VAL_DIR)
    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_gen = val_aug.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
else:
    print("No separate val folder found. Using validation_split from training data.")
    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    val_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

num_classes_detected = len(train_gen.class_indices)
print("Detected classes from folders:", num_classes_detected)
if num_classes_detected != NUM_CLASSES:
    print(f"WARNING: expected NUM_CLASSES={NUM_CLASSES} but found {num_classes_detected} folders. Continuing with {num_classes_detected}.")
    NUM_CLASSES = num_classes_detected

# -------------------------
# build model
# -------------------------
print("==> Building model")
model = Alex_model(out_dims=NUM_CLASSES, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))

opt = SGD(learning_rate=LR_INIT, momentum=MOMENTUM, nesterov=False)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -------------------------
# callbacks
# -------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", "alexnet_" + timestamp)
os.makedirs(log_dir, exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    OUTPUT_WEIGHTS,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)
tensorboard_cb = TensorBoard(log_dir=log_dir)

def lr_schedule(epoch):
    # step decay: reduce by 0.1 every 20 epochs
    drop = 0.1 ** (epoch // 20)
    return LR_INIT * drop

lr_cb = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint_cb, tensorboard_cb, lr_cb]

# -------------------------
# training
# -------------------------
steps_per_epoch = math.ceil(train_gen.samples / BATCH_SIZE)
validation_steps = math.ceil(val_gen.samples / BATCH_SIZE)

print(f"Training for {EPOCHS} epochs, steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

# START TIMER
start = time.time()

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# ⏱ END TIMER
end = time.time()

# save final weights
print("Saving final weights to:", OUTPUT_WEIGHTS)
model.save_weights(OUTPUT_WEIGHTS)

elapsed = end - start
print(f"Training complete. Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes, {elapsed/3600:.2f} hours)")