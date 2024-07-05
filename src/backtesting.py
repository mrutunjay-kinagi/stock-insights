import pandas as pd

class Backtester:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds

    def apply_thresholds(self, data):
        conditions = []
        if 'trailingPE' in data.columns and 'PE_ratio' in self.thresholds:
            conditions.append(data['trailingPE'] <= self.thresholds['PE_ratio'])
        if 'priceToBook' in data.columns and 'PB_ratio' in self.thresholds:
            conditions.append(data['priceToBook'] <= self.thresholds['PB_ratio'])
        if 'debtToEquity' in data.columns and 'de_ratio' in self.thresholds:
            conditions.append(data['debtToEquity'] <= self.thresholds['de_ratio'])
        if 'earningsGrowth' in data.columns and 'earnings_growth' in self.thresholds:
            conditions.append(data['earningsGrowth'] >= self.thresholds['earnings_growth'])
        if 'revenueGrowth' in data.columns and 'revenue_growth' in self.thresholds:
            conditions.append(data['revenueGrowth'] >= self.thresholds['revenue_growth'])

        if conditions:
            combined_conditions = conditions.pop()
            for condition in conditions:
                combined_conditions &= condition
            data['Buy_Signal'] = combined_conditions
        else:
            data['Buy_Signal'] = False

        return data

    def backtest(self, data):
        data = self.apply_thresholds(data)
        return data

    def evaluate_strategy(self, data):
        backtested_data = self.backtest(data)
        accuracy = self.calculate_accuracy(backtested_data)
        return accuracy

    def calculate_accuracy(self, data):
        # Calculate accuracy as the percentage of correct Buy/Sell signals
        total_signals = len(data)
        correct_signals = len(data[data['Buy_Signal'] == data['actual']])  # Assuming 'actual' column is present
        accuracy = correct_signals / total_signals if total_signals else 0
        return accuracy
