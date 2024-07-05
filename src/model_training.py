from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

class ModelTrainer:
    def __init__(self):
        self.classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regression_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_classification_model(self, X_train, y_train):
        self.classification_model.fit(X_train, y_train)

    def evaluate_classification_model(self, X_test, y_test):
        predictions = self.classification_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        return accuracy, report

    def train_regression_model(self, X_train, y_train):
        self.regression_model.fit(X_train, y_train)

    def evaluate_regression_model(self, X_test, y_test):
        predictions = self.regression_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        return mse, predictions
