import json

def load_thresholds(file_path):
    with open(file_path, 'r') as file:
        thresholds = json.load(file)
    return thresholds
