#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ------------------------------
# CONFIGURATION
# ------------------------------
train_path = r"D:\Stuff\FRC\Chinese character recognition\Deep_Learning-master\train_data\train"
output_file = "class_map.txt"

# ------------------------------
# MAIN
# ------------------------------
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Could not find training folder: {train_path}")

# Get sorted list of subfolders (each is a class)
classes = sorted(next(os.walk(train_path))[1])
print(f"Found {len(classes)} classes in {train_path}")

# Write mapping to file
with open(output_file, "w", encoding="utf-8") as f:
    for idx, cls_name in enumerate(classes):
        f.write(f"class {idx} = {cls_name}\n")

print(f"Class map saved to: {output_file}")
