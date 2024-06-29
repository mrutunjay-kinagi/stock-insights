from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report
