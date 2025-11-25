import os
import shutil
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization, PReLU, Dense, Dropout, Input
from tensorflow.keras.activations import softmax
from PIL import Image

# -----------------------------
VAL_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train_val\train_val"
SORTED_VAL_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\val_sorted"
TRAIN_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
WEIGHTS_PATH = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\interface\Weights_Alex_Net.h5"
IMG_SIZE = (128,128)  # match your AlexNet training

# -----------------------------
# Custom AlexNet model
def bn_relu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def Alex_model(out_dims, input_shape=(128,128,1)):
    input_dim = Input(input_shape)
    x = Conv2D(96,(20,20),strides=(2,2),padding='valid')(input_dim)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    x = Conv2D(256,(5,5),padding='same')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    x = Conv2D(384,(3,3),padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(384,(3,3),padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(256,(3,3),padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Dropout(0.2)(x)
    x = Dense(4096)(x)
    x = Dropout(0.25)(x)
    x = Dense(out_dims)(x)
    output = Activation('softmax')(x)
    return Model(inputs=input_dim, outputs=output)

# -----------------------------
# Load model
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
NUM_CLASSES = len(class_names)
model = Alex_model(NUM_CLASSES)
model.load_weights(WEIGHTS_PATH)
print(f"✅ AlexNet loaded with {NUM_CLASSES} classes.")

# -----------------------------
# Make class folders
for cname in class_names:
    os.makedirs(os.path.join(SORTED_VAL_DIR, cname), exist_ok=True)

# -----------------------------
# Preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)/255.0
    arr = np.expand_dims(arr, axis=(0,-1))  # shape: (1,H,W,1)
    return arr

# -----------------------------
# Sort images
val_files = [f for f in os.listdir(VAL_DIR) if os.path.isfile(os.path.join(VAL_DIR,f))]
print(f"Found {len(val_files)} images to sort.")

for f in val_files:
    path = os.path.join(VAL_DIR,f)
    try:
        img_arr = preprocess_image(path)
        preds = model.predict(img_arr)
        idx = int(np.argmax(preds))
        cname = class_names[idx]
        dest = os.path.join(SORTED_VAL_DIR,cname,f)
        shutil.move(path,dest)
        print(f"{f} -> {cname} | confidence={preds[0,idx]:.4f}")
    except Exception as e:
        print(f"Skipping {f} due to error: {e}")

print("✅ Validation images sorted successfully.")