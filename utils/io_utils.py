# utils/io_utils.py
import os
import json


def ensure_dir(path):
os.makedirs(path, exist_ok=True)


def read_json(path):
with open(path, 'r', encoding='utf-8') as f:
return json.load(f)


def write_json(path, obj):
with open(path, 'w', encoding='utf-8') as f:
json.dump(obj, f, indent=2)
