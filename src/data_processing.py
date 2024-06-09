import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_files):
        self.data_files = data_files

    def load_and_process_data(self):
        data_frames = [pd.read_csv(file) for file in self.data_files]
        data = pd.concat(data_frames, ignore_index=True)
        data = self.feature_engineering(data)
        return data

    def feature_engineering(self, data):
        data['PE_Ratio'] = data['Close'] / data['EPS']
        data['ROE'] = data['NetIncome'] / data['TotalEquity']
        data['PB_Ratio'] = data['Close'] / data['BookValue']
        data['Dividend_Yield'] = data['Dividend'] / data['Close']
        data['Debt_to_Equity'] = data['TotalDebt'] / data['TotalEquity']

        data['recommendation'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        data.to_csv('./data/processed/processed_data.csv', index=False)
        return data

    def split_data(self, data):
        X = data.drop(columns=['recommendation'])
        y = data['recommendation']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    files = ['../data/raw/RELIANCE.NS.csv', '../data/raw/TCS.NS.csv', '../data/raw/INFY.NS.csv']
    processor = DataProcessor(files)
    data = processor.load_and_process_data()
    data.to_csv('../data/processed/processed_data.csv', index=False)
    X_train, X_test, y_train, y_test = processor.split_data(data)
