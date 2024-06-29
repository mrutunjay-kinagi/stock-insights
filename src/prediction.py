import pandas as pd

class Predictor:
    def __init__(self, model):
        self.model = model

    def make_prediction(self, data):
        prediction = self.model.predict(data)
        return prediction
