import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, fundamental_data, technical_data):
        self.fundamental_data = fundamental_data
        self.technical_data = technical_data
        self.scaler = StandardScaler()

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

        # Fit and transform the scaler on the training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_latest_data_point(self, data):
        latest_data_point = data.iloc[-1:].copy()  # Get the last row
        return latest_data_point

    def scale_latest_data(self, latest_data_point):
        latest_data_point_scaled = self.scaler.transform(latest_data_point)
        return latest_data_point_scaled
