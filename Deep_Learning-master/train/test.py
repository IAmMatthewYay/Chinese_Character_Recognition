import os
import shutil
import cv2
from tqdm import tqdm
from roboflow import Roboflow

# ==== CONFIG ====
TRAIN_DIR = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
TEMP_DIR = os.path.join(os.path.dirname(TRAIN_DIR), "temp_upload")
PROJECT_NAME = "chinese-charecter-classification-16jum"
WORKSPACE = "matthew-oeagp"
API_KEY = "bPu0jxCTIekRcjqReO9t"

# ==== CLASS MAPPING ====
CLASS_MAPPING = {
    1: "且", 2: "世", 3: "东", 4: "九", 5: "亭", 6: "今", 7: "从", 8: "令", 9: "作", 10: "使",
    11: "侯", 12: "元", 13: "光", 14: "利", 15: "印", 16: "去", 17: "受", 18: "右", 19: "司", 20: "合",
    21: "名", 22: "周", 23: "命", 24: "和", 25: "唯", 26: "堂", 27: "士", 28: "多", 29: "夜", 30: "奉",
    31: "女", 32: "好", 33: "始", 34: "字", 35: "孝", 36: "守", 37: "宗", 38: "官", 39: "定", 40: "宜",
    41: "室", 42: "家", 43: "寒", 44: "左", 45: "常", 46: "建", 47: "徐", 48: "御", 49: "必", 50: "思",
    51: "意", 52: "我", 53: "敬", 54: "新", 55: "易", 56: "春", 57: "更", 58: "朝", 59: "李", 60: "来",
    61: "林", 62: "正", 63: "武", 64: "氏", 65: "永", 66: "流", 67: "海", 68: "深", 69: "清", 70: "游",
    71: "父", 72: "物", 73: "玉", 74: "用", 75: "申", 76: "白", 77: "皇", 78: "益", 79: "福", 80: "秋",
    81: "立", 82: "章", 83: "老", 84: "臣", 85: "良", 86: "莫", 87: "虎", 88: "衣", 89: "西", 90: "起",
    91: "足", 92: "身", 93: "通", 94: "遂", 95: "重", 96: "陵", 97: "雨", 98: "高", 99: "黄", 100: "鼎"
}

# ==== INIT ROBOFLOW ====
rf = Roboflow(api_key=API_KEY)
print("Loading Roboflow workspace...")
workspace = rf.workspace(WORKSPACE)
print("Loading Roboflow project...")
project = workspace.project(PROJECT_NAME)

# ==== PREPARE TEMP FOLDERS ====
os.makedirs(TEMP_DIR, exist_ok=True)
existing_temp = sorted(
    [d for d in os.listdir(TEMP_DIR) if os.path.isdir(os.path.join(TEMP_DIR, d)) and d.startswith("class_")],
    key=lambda x: int(x.split("_")[1])
)

if existing_temp:
    start_class = int(existing_temp[0].split("_")[1])
else:
    start_class = min(CLASS_MAPPING.keys())

# Only prepare folders that don’t already exist
for idx, chinese_name in CLASS_MAPPING.items():
    if idx < start_class:
        continue
    class_temp = os.path.join(TEMP_DIR, f"class_{idx}")
    src_folder = os.path.join(TRAIN_DIR, chinese_name)
    if not os.path.exists(class_temp):
        os.makedirs(class_temp, exist_ok=True)
        if os.path.isdir(src_folder):
            for file in os.listdir(src_folder):
                src_path = os.path.join(src_folder, file)
                dst_path = os.path.join(class_temp, file)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

print("All temp folders prepared!")

# ==== UPLOAD LOOP ====
failed_uploads = []
total_classes = len(CLASS_MAPPING)

for idx, chinese_name in sorted(CLASS_MAPPING.items()):
    class_temp = os.path.join(TEMP_DIR, f"class_{idx}")
    if not os.path.isdir(class_temp) or not os.listdir(class_temp):
        continue  # skip if no temp folder or empty

    batch_name = f"Batch_class_{idx}"
    files = os.listdir(class_temp)

    print(f"\n=== Starting upload session {idx}/{total_classes} ===")
    print(f"Class: {chinese_name} (class_{idx}) | Batch: {batch_name}")

    for file_name in tqdm(files, desc=f"Uploading {chinese_name}", unit="img"):
        file_path = os.path.join(class_temp, file_name)
        img = cv2.imread(file_path)
        if img is None:
            print(f"[SKIPPED] Unreadable file: {file_path}")
            failed_uploads.append(file_path)
            continue

        success = False
        for attempt in range(3):
            try:
                project.upload(
                    image_path=file_path,
                    split="train",
                    batch_name=batch_name
                )
                print(f"[OK] Uploaded: {file_name}")
                success = True
                break
            except Exception as e:
                print(f"[WARN] Upload failed ({attempt+1}/3) for {file_name}: {e}")
        if not success:
            print(f"[SKIPPED] Failed after 3 attempts: {file_path}")
            failed_uploads.append(file_path)

    # Delete temp folder after class upload
    shutil.rmtree(class_temp)
    print(f"Class {chinese_name} upload complete, temp folder cleared.\n")

# ==== FINAL SUMMARY ====
if failed_uploads:
    print("\nThe following files failed to upload:")
    for f in failed_uploads:
        print(" -", f)
else:
    print("\nAll uploads completed successfully!")