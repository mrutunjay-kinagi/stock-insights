import json

def load_thresholds(filepath):
    with open(filepath, 'r') as file:
        thresholds = json.load(file)
    return thresholds
