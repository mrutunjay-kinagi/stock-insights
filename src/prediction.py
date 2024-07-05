class Predictor:
    def __init__(self, model):
        self.model = model

    def make_prediction(self, data):
        return self.model.predict(data)
