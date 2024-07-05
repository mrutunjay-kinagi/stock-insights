import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, fundamental_data, technical_data):
        self.fundamental_data = fundamental_data
        self.technical_data = technical_data

    def preprocess(self):
        self.technical_data.dropna(inplace=True)
        if 'Adj Close' in self.technical_data.columns:
            self.technical_data.drop(columns=['Adj Close'], inplace=True)
        self.technical_data['Target'] = (self.technical_data['Close'].shift(-1) > self.technical_data['Close']).astype(int)
        return self.technical_data.dropna()

    def split_data(self, data):
        X = data.drop(columns=['Target', 'Close'])
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
