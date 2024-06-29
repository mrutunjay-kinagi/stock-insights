from src.data_collection import DataCollector
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.prediction import Predictor
from src.backtesting import Backtester
from src.utils import load_thresholds

def main():
    # Input from user
    ticker = input("Enter the stock ticker: ")
    print("Select the type of threshold:")
    print("1. Conservative")
    print("2. Moderate")
    print("3. Aggressive")
    threshold_option = input("Enter the number corresponding to your choice: ")

    threshold_mapping = {
        "1": "Conservative",
        "2": "Moderate",
        "3": "Aggressive"
    }

    selected_threshold_type = threshold_mapping.get(threshold_option, "Conservative")

    # Load thresholds
    thresholds = load_thresholds('thresholds.json')
    selected_thresholds = thresholds.get(selected_threshold_type)

    # Data collection
    collector = DataCollector(ticker)
    fundamental_data, technical_data = collector.get_data()
    

    # Data processing
    processor = DataProcessor(fundamental_data, technical_data)
    data = processor.preprocess()
    X_train, X_test, y_train, y_test = processor.split_data(data)

    # Model training
    trainer = ModelTrainer()
    trainer.train_model(X_train, y_train)
    report = trainer.evaluate_model(X_test, y_test)
    print("Model Evaluation Report:")
    print(report)

    # Prediction
    predictor = Predictor(trainer.model)
    prediction = predictor.make_prediction(X_test)
    print("Prediction:")
    print(prediction)

    # Decision based on thresholds
    decision = "Buy" if prediction[-1] else "Sell"
    print(f"Recommendation based on {selected_threshold_type} threshold: {decision}")

    # Backtesting
    backtester = Backtester(trainer.model, selected_thresholds)
    accuracy, final_balance = backtester.evaluate_strategy(data)
    print(f"Backtesting Accuracy: {accuracy}")
    print(f"Final Balance after Backtesting: {final_balance}")

if __name__ == "__main__":
    main()
