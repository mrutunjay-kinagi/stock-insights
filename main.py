from src.data_collection import DataCollector
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.prediction import Predictor
from src.backtesting import Backtester
import pandas as pd

def get_user_thresholds():
    threshold_options = {
        'conservative': {
            "pe_threshold": 15,
            "roe_threshold": 15,
            "pb_threshold": 1.5,
            "dividend_yield_threshold": 3,
            "debt_to_equity_threshold": 0.5
        },
        'balanced': {
            "pe_threshold": 20,
            "roe_threshold": 12,
            "pb_threshold": 2,
            "dividend_yield_threshold": 2,
            "debt_to_equity_threshold": 1
        },
        'aggressive': {
            "pe_threshold": 25,
            "roe_threshold": 10,
            "pb_threshold": 3,
            "dividend_yield_threshold": 1,
            "debt_to_equity_threshold": 2
        }
    }

    print("Choose your investment strategy:")
    print("1. Conservative")
    print("2. Balanced")
    print("3. Aggressive")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        return threshold_options['conservative']
    elif choice == '2':
        return threshold_options['balanced']
    elif choice == '3':
        return threshold_options['aggressive']
    else:
        print("Invalid choice. Defaulting to Balanced strategy.")
        return threshold_options['balanced']

def main():
    # Collect data
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    collector = DataCollector(tickers)
    data = collector.fetch_data('2020-01-01', '2023-01-01')
    collector.save_data(data, './data/raw')

    # Process data
    files = ['./data/raw/RELIANCE.NS.csv', './data/raw/TCS.NS.csv', './data/raw/INFY.NS.csv']
    processor = DataProcessor(files)
    data = processor.load_and_process_data()
    data.to_csv('./data/processed/processed_data.csv', index=False)
    X_train, X_test, y_train, y_test = processor.split_data(data)

    # Train model
    trainer = ModelTrainer()
    model = trainer.train_model(X_train, y_train)
    mse = trainer.evaluate_model(model, X_test, y_test)
    print(f"Model Mean Squared Error: {mse}")
    trainer.save_model(model, './models/trained_model.pkl')


    # Get user thresholds
    thresholds = get_user_thresholds()

    # Backtesting
    backtester = Backtester('./models/trained_model.pkl', './data/processed/processed_data.csv', thresholds)
    recommendations = backtester.apply_strategy()
    metrics = backtester.evaluate_strategy(recommendations)
    print("Backtesting Results:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")

    # Predict and give recommendation
    predictor = Predictor('./models/trained_model.pkl')
    new_data = pd.read_csv('./data/processed/processed_data.csv')
    predictions = predictor.predict(new_data)

    recommendations = []
    for i, prediction in enumerate(predictions):
        reasons = []
        if prediction['PE_Ratio'] <= thresholds['pe_threshold']:
            reasons.append(f"P/E Ratio ({prediction['PE_Ratio']}) is below the threshold ({thresholds['pe_threshold']})")
        if prediction['ROE'] >= thresholds['roe_threshold']:
            reasons.append(f"ROE ({prediction['ROE']}) is above the threshold ({thresholds['roe_threshold']})")
        if prediction['PB_Ratio'] <= thresholds['pb_threshold']:
            reasons.append(f"P/B Ratio ({prediction['PB_Ratio']}) is below the threshold ({thresholds['pb_threshold']})")
        if prediction['Dividend_Yield'] >= thresholds['dividend_yield_threshold']:
            reasons.append(f"Dividend Yield ({prediction['Dividend_Yield']}) is above the threshold ({thresholds['dividend_yield_threshold']})")
        if prediction['Debt_to_Equity'] <= thresholds['debt_to_equity_threshold']:
            reasons.append(f"Debt-to-Equity Ratio ({prediction['Debt_to_Equity']}) is below the threshold ({thresholds['debt_to_equity_threshold']})")

        if len(reasons) == 5:
            recommendations.append(f"Stock {new_data['ticker'][i]} is a good buy. Reasons: " + ", ".join(reasons))
        else:
            recommendations.append(f"Stock {new_data['ticker'][i]} is not a good buy. Reasons: " + ", ".join(reasons))

    for recommendation in recommendations:
        print(recommendation)

if __name__ == "__main__":
    main()
