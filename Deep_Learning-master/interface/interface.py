#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, base64, threading, numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Activation, Flatten, BatchNormalization,
    PReLU, Dense, Dropout, Input, Add, GlobalAveragePooling2D
)

# =========================================================
# Configuration
# =========================================================
UPLOAD_FOLDER = "static/uploads"
LOGO_FOLDER = "static/logos"
MODEL_FOLDER = "models"
CERT_FOLDER = "certs"
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGO_FOLDER, exist_ok=True)
os.makedirs(CERT_FOLDER, exist_ok=True)

# =========================================================
# Load class map
# =========================================================
class_map = {}
class_map_path = os.path.join(LOGO_FOLDER, "class_map.txt")
if os.path.exists(class_map_path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("=",1)
            try:
                idx = int(parts[0].replace("class","").strip())
                char = parts[1].strip()
                class_map[idx] = char
            except:
                continue
else:
    print("⚠️ class_map.txt not found — predictions will show indices only.")

# =========================================================
# Model definitions
# =========================================================

def bn_relu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

# ------------------ AlexNet ------------------
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

# ------------------ ResNet-18 ------------------
def res_block(x, filters, stride=1):
    shortcut = x

    x = Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), strides=stride, use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet_model(out_dims=100, input_shape=(128,128,1)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64,(7,7),strides=2,padding='same',use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    x = res_block(x, 64)
    x = res_block(x, 64)

    x = res_block(x, 128, stride=2)
    x = res_block(x, 128)

    x = res_block(x, 256, stride=2)
    x = res_block(x, 256)

    x = res_block(x, 512, stride=2)
    x = res_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(out_dims, activation='softmax')(x)

    return Model(inputs, outputs, name="ResNet18_Custom")


# ------------------ ResNet-101 ------------------
def bottleneck_block(x, filters, stride=1):
    shortcut = x

    x = Conv2D(filters, (1,1), strides=stride, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3,3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters*4, (1,1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters*4:
        shortcut = Conv2D(filters*4, (1,1), strides=stride, use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def ResNet101_model(out_dims=100, input_shape=(128,128,1)):
    inputs = Input(input_shape)

    x = Conv2D(64,(7,7),strides=2,padding="same",use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding="same")(x)

    for _ in range(3):
        x = bottleneck_block(x, 64)

    x = bottleneck_block(x, 128, stride=2)
    for _ in range(3):
        x = bottleneck_block(x, 128)

    x = bottleneck_block(x, 256, stride=2)
    for _ in range(22):
        x = bottleneck_block(x, 256)

    x = bottleneck_block(x, 512, stride=2)
    for _ in range(2):
        x = bottleneck_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(out_dims, activation="softmax")(x)

    return Model(inputs, out, name="ResNet101")


# =========================================================
# Model loader
# =========================================================
current_model_name = None
model = None

def load_model_by_name(name):
    global model, current_model_name

    try:
        if name == "alexnet":
            model = Alex_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Alex_Net.h5")
            current_model_name = "alexnet"
            print("✅ Loaded AlexNet")

        elif name == "alexnet_old":
            model = Alex_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Alex_Net_Old.h5")
            current_model_name = "alexnet_old"
            print("✅ Loaded AlexNet Old")

        elif name == "alexnet_old2":
            model = Alex_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Alex_Net_Old_2.h5")
            current_model_name = "alexnet_old2"
            print("✅ Loaded AlexNet Old 2")

        elif name == "resnet18":
            model = ResNet_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_18.h5")
            current_model_name = "resnet18"
            print("✅ Loaded ResNet18")

        elif name == "resnet18_old":
            model = ResNet_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_18_Old.h5")
            current_model_name = "resnet18_old"
            print("✅ Loaded ResNet18 Old")

        elif name == "resnet18_old2":
            model = ResNet_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_18_Old_2.h5")
            current_model_name = "resnet18_old2"
            print("✅ Loaded ResNet18 Old 2")

        elif name == "resnet18_old3":
            model = ResNet_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_18_Old_3.h5")
            current_model_name = "resnet18_old3"
            print("✅ Loaded ResNet18 Old 3")

        elif name == "resnet101":
            model = ResNet101_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_101.h5")
            current_model_name = "resnet101"
            print("✅ Loaded ResNet101")
        
        elif name == "resnet101_old":
            model = ResNet101_model(out_dims=100)
            model.load_weights(f"{MODEL_FOLDER}/Weights_Resnet_101_Old.h5")
            current_model_name = "resnet101_old"
            print("✅ Loaded ResNet101")

        else:
            raise ValueError("Unknown model!")

    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")


# Load default model
load_model_by_name("alexnet")

# =========================================================
# Preprocessing + Prediction
# =========================================================
def preprocess_image(img_path_or_PIL):
    if isinstance(img_path_or_PIL, str):
        img = Image.open(img_path_or_PIL).convert("L").resize(IMG_SIZE)
    else:
        img = img_path_or_PIL.convert("L").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array


def predict_image(img_path_or_PIL):
    img_array = preprocess_image(img_path_or_PIL)
    preds = model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(preds[0][class_index])
    char = class_map.get(class_index, f"class_{class_index}")
    return f"class_{class_index} | {char}", confidence * 100

# =========================================================
# Flask App
# =========================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET","POST"])
def index():
    uploaded_img = None
    result = None
    confidence = None
    global current_model_name

    if request.method == "POST":
        selected_model = request.form.get("model_select")
        if selected_model and selected_model != current_model_name:
            load_model_by_name(selected_model)

        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                uploaded_img = url_for("static", filename=f"uploads/{filename}")
                result, confidence = predict_image(filepath)

    logos = [
        {"path":"First_Logo.jpg","alt":"First Logo Missing"},
        {"path":"Unearthed_Logo.png","alt":"Unearthed Logo Missing"},
        {"path":"Second_Logo.png","alt":"Second Logo Missing"},
        {"path":"Image.png","alt":"Third Logo Missing"},
    ]

    return render_template("index.html",
                           uploaded_img=uploaded_img,
                           result=result,
                           confidence=f"{confidence:.2f}%" if confidence else None,
                           logos=logos,
                           current_model=current_model_name)

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    try:
        data = request.get_json()
        img_data = re.sub('^data:image/.+;base64,','',data['image'])
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        result, confidence = predict_image(img)
        return jsonify({"result": result, "confidence": f"{confidence:.2f}"})
    except Exception as e:
        return jsonify({"result":"Error","confidence":"0","error":str(e)})


# =========================================================
# HTTP + HTTPS servers
# =========================================================
def run_http():
    app.run(host="0.0.0.0", port=5000, debug=False)

def run_https():
    cert_file = os.path.join(CERT_FOLDER, "cert.pem")
    key_file = os.path.join(CERT_FOLDER, "key.pem")
    if os.path.exists(cert_file) and os.path.exists(key_file):
        app.run(host="0.0.0.0", port=5443, debug=False, ssl_context=(cert_file, key_file))
    else:
        print("⚠️ HTTPS certificates not found — skipping HTTPS startup.")
        
if __name__ == "__main__":
    print("🔹 HTTP:  http://<your_lan_ip>:5000")
    print("🔹 HTTPS: https://<your_lan_ip>:5443")
    threading.Thread(target=run_http).start()
    threading.Thread(target=run_https).start()