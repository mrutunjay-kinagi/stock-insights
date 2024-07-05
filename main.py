from src.data_collection import DataCollector
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.prediction import Predictor
from src.backtesting import Backtester
from src.utils import load_thresholds
import os

def main():
    # Input from user
    ticker = input("Enter the stock ticker: ")
    print("Select the type of analysis:")
    print("1. Fundamental Analysis")
    print("2. Technical Analysis")
    analysis_option = input("Enter the number corresponding to your choice: ")

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

    os.makedirs('data/raw', exist_ok=True)

    # Save data to CSV files in data/raw folder
    fundamental_data.to_csv(f'data/raw/{ticker}_fundamental_data.csv', index=False)
    technical_data.to_csv(f'data/raw/{ticker}_technical_data.csv')

    if fundamental_data.empty or technical_data.empty:
        print("Failed to fetch data for the ticker.")
        return

    if analysis_option == "1":
        # Fundamental analysis
        print("Fundamental Analysis:")
        print(fundamental_data)

        # Define the metrics to be used for fundamental analysis
        metrics = {
            'trailingPE': ('PE Ratio', 'PE_ratio'),
            'priceToBook': ('PB Ratio', 'PB_ratio'),
            'debtToEquity': ('Debt to Equity Ratio', 'de_ratio'),
            'earningsGrowth': ('Earnings Growth', 'earnings_growth'),
            'revenueGrowth': ('Revenue Growth', 'revenue_growth')
        }

        recommendations = []

        for key, (metric_name, threshold_key) in metrics.items():
            if key in fundamental_data.columns:
                value = fundamental_data[key].values[0]
                threshold = selected_thresholds.get(threshold_key)
                if threshold is not None:
                    decision = "Buy" if value and value < threshold else "Sell"
                    reason = f"{metric_name} ({value}) is {'less' if decision == 'Buy' else 'greater'} than {threshold}"
                    recommendations.append((decision, reason))

        if recommendations:
            final_decision = "Buy" if all(rec[0] == "Buy" for rec in recommendations) else "Sell"
            print(f"Overall Recommendation based on {selected_threshold_type} threshold: {final_decision}")
            for rec in recommendations:
                print(f"Reason: {rec[1]}")
        else:
            print("No sufficient data for a comprehensive fundamental analysis.")

    elif analysis_option == "2":
        # Technical analysis
        processor = DataProcessor(fundamental_data, technical_data)
        data = processor.preprocess()

        if data.empty:
            print("No data available after preprocessing.")
            return

        X_train, X_test, y_train, y_test = processor.split_data(data)

        # Model training
        trainer = ModelTrainer()
        trainer.train_classification_model(X_train, y_train)
        accuracy, report = trainer.evaluate_classification_model(X_test, y_test)
        print("Model Evaluation Report:")
        print(report)

        # Price Prediction
        trainer.train_regression_model(X_train, y_train)
        mse, predicted_prices = trainer.evaluate_regression_model(X_test, y_test)
        print("Mean Squared Error:", mse)
        print("Predicted Prices:")
        print(predicted_prices)

        # Prediction
        predictor = Predictor(trainer.classification_model)
        prediction = predictor.make_prediction(X_test)
        print("Prediction:")
        print(prediction)

        # Decision based on thresholds
        decision = "Buy" if prediction[-1] else "Sell"
        print(f"Recommendation based on {selected_threshold_type} threshold: {decision}")

        # Backtesting
        #backtester = Backtester(trainer.classification_model, selected_thresholds)
        #accuracy, final_balance = backtester.evaluate_strategy(data)
        #print(f"Backtesting Accuracy: {accuracy}")
    else:
        print("Invalid option selected.")

if __name__ == "__main__":
    main()
