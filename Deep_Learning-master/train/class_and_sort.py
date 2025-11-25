import requests
import re
import time

# ==== CONFIG ====
API_KEY = "bPu0jxCTIekRcjqReO9t"
WORKSPACE = "matthew-oeagp"
PROJECT_NAME = "chinese-charecter-classification-16jum"

BASE_URL = f"https://api.roboflow.com/projects/{WORKSPACE}/{PROJECT_NAME}"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def fetch_batches():
    url = f"{BASE_URL}/batches"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["batches"]

def create_label(label_name):
    url = f"{BASE_URL}/labels"
    data = {"name": label_name}
    resp = requests.post(url, headers=HEADERS, json=data)
    if resp.status_code == 200:
        print(f"[OK] Created label {label_name}")
    elif resp.status_code == 409:
        print(f"[SKIP] Label {label_name} already exists")
    else:
        print(f"[FAIL] Failed to create label {label_name}: {resp.text}")

def assign_images_to_label(batch_id, label_name):
    url = f"{BASE_URL}/batches/{batch_id}/annotate"
    data = {"label": label_name}
    resp = requests.post(url, headers=HEADERS, json=data)
    if resp.status_code == 200:
        print(f"[OK] Annotated Batch {batch_id} → {label_name}")
    else:
        print(f"[FAIL] Failed to annotate Batch {batch_id}: {resp.text}")

def main():
    print("Fetching batches...")
    batches = fetch_batches()

    for batch in batches:
        batch_name = batch["name"]

        # Only process batches that follow "Batch_class_X" naming
        match = re.match(r"Batch_class_(\d+)", batch_name, re.IGNORECASE)
        if not match:
            print(f"[SKIP] Ignoring batch {batch_name} (no class number)")
            continue

        label_name = f"Class_{match.group(1)}"
        batch_id = batch["id"]

        create_label(label_name)
        assign_images_to_label(batch_id, label_name)

        time.sleep(0.5)  # avoid hitting rate limits

if __name__ == "__main__":
    main()