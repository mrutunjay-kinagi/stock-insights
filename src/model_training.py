from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def save_model(self, model, file_path):
        joblib.dump(model, file_path)

if __name__ == "__main__":
    from data_processing import DataProcessor

    files = ['../data/raw/RELIANCE.NS.csv', '../data/raw/TCS.NS.csv', '../data/raw/INFY.NS.csv']
    processor = DataProcessor(files)
    data = processor.load_and_process_data()
    X_train, X_test, y_train, y_test = processor.split_data(data)

    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train)
    mse = trainer.evaluate_model(model, X_test, y_test)
    print(f"Model Mean Squared Error: {mse}")
    trainer.save_model(model, '../models/trained_model.pkl')
