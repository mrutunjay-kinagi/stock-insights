import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_file):
        self.model = joblib.load(model_file)

    def preprocess_new_data(self, new_data):
        new_data['PE_Ratio'] = new_data['Close'] / new_data['EPS']
        new_data['ROE'] = new_data['NetIncome'] / new_data['TotalEquity']
        new_data['PB_Ratio'] = new_data['Close'] / new_data['BookValue']
        new_data['Dividend_Yield'] = new_data['Dividend'] / new_data['Close']
        new_data['Debt_to_Equity'] = new_data['TotalDebt'] / new_data['TotalEquity']
        return new_data

    def predict(self, new_data):
        processed_data = self.preprocess_new_data(new_data)
        predictions = self.model.predict(processed_data)
        processed_data['Predictions'] = predictions
        return processed_data.to_dict('records')

