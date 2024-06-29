import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, fundamental_data, technical_data):
        self.fundamental_data = fundamental_data
        self.technical_data = technical_data

    def preprocess(self):
        # Combine fundamental and technical data
        data = pd.merge(self.technical_data, self.fundamental_data, left_index=True, right_index=True)
        print("Technical and Fundamental data: ")
        print(self.technical_data)
        print(self.fundamental_data)
        print("Data: ")
        print(data)
        
        # Drop rows with missing values
        data.dropna(inplace=True)

        print("Data after DropNA: ")
        print(data)

        # Create a target variable (e.g., 'recommendation' based on some logic or existing column)
        data['recommendation'] = data['Close'].pct_change().shift(-1) > 0
        
        return data

    def split_data(self, data):
        X = data.drop(columns=['recommendation'])
        y = data['recommendation']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
