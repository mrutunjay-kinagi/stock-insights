import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.prediction import Predictor

class Backtester:
    def __init__(self, model_file, data_file, thresholds):
        self.predictor = Predictor(model_file)
        self.data = pd.read_csv(data_file)
        self.thresholds = thresholds

    def apply_strategy(self):
        predictions = self.predictor.predict(self.data)
        recommendations = []

        for i, prediction in enumerate(predictions):
            reasons = []
            if prediction['PE_Ratio'] <= self.thresholds['pe_threshold']:
                reasons.append(f"P/E Ratio ({prediction['PE_Ratio']}) is below the threshold ({self.thresholds['pe_threshold']})")
            if prediction['ROE'] >= self.thresholds['roe_threshold']:
                reasons.append(f"ROE ({prediction['ROE']}) is above the threshold ({self.thresholds['roe_threshold']})")
            if prediction['PB_Ratio'] <= self.thresholds['pb_threshold']:
                reasons.append(f"P/B Ratio ({prediction['PB_Ratio']}) is below the threshold ({self.thresholds['pb_threshold']})")
            if prediction['Dividend_Yield'] >= self.thresholds['dividend_yield_threshold']:
                reasons.append(f"Dividend Yield ({prediction['Dividend_Yield']}) is above the threshold ({self.thresholds['dividend_yield_threshold']})")
            if prediction['Debt_to_Equity'] <= self.thresholds['debt_to_equity_threshold']:
                reasons.append(f"Debt-to-Equity Ratio ({prediction['Debt_to_Equity']}) is below the threshold ({self.thresholds['debt_to_equity_threshold']})")

            if len(reasons) == 5:
                recommendations.append(1)  # Buy
            else:
                recommendations.append(0)  # Don't Buy

        return recommendations

    def evaluate_strategy(self, recommendations):
        actuals = self.data['target_column']  # Assuming 'target_column' contains the actual buy/sell decisions
        accuracy = accuracy_score(actuals, recommendations)
        precision = precision_score(actuals, recommendations)
        recall = recall_score(actuals, recommendations)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

if __name__ == "__main__":
    thresholds = {
        "pe_threshold": 20,
        "roe_threshold": 12,
        "pb_threshold": 2,
        "dividend_yield_threshold": 2,
        "debt_to_equity_threshold": 1
    }

    backtester = Backtester('./models/trained_model.pkl', './data/processed/processed_data.csv', thresholds)
    recommendations = backtester.apply_strategy()
    metrics = backtester.evaluate_strategy(recommendations)

    print("Backtesting Results:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
