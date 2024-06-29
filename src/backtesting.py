import pandas as pd
from sklearn.metrics import accuracy_score

class Backtester:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds

    def apply_thresholds(self, data):
        conditions = (
            (data['PE Ratio'] <= self.thresholds['PE_ratio']) &
            (data['ROE'] >= self.thresholds['ROE']) &
            (data['Dividend Yield'] >= self.thresholds['Dividend_Yield']) &
            (data['DE Ratio'] <= self.thresholds['DE_ratio'])
        )
        data['recommendation'] = conditions
        return data

    def backtest(self, data):
        data = self.apply_thresholds(data)
        predictions = self.model.predict(data.drop(columns=['recommendation']))
        accuracy = accuracy_score(data['recommendation'], predictions)
        return accuracy, data

    def simulate_trades(self, data, initial_balance=10000):
        balance = initial_balance
        shares = 0
        for i in range(len(data) - 1):
            if data['recommendation'].iloc[i]:
                # Buy signal
                shares_to_buy = balance // data['Close'].iloc[i]
                balance -= shares_to_buy * data['Close'].iloc[i]
                shares += shares_to_buy
            elif shares > 0:
                # Sell signal
                balance += shares * data['Close'].iloc[i]
                shares = 0
        # Final liquidation
        if shares > 0:
            balance += shares * data['Close'].iloc[-1]
        return balance

    def evaluate_strategy(self, data, initial_balance=10000):
        accuracy, data_with_recommendations = self.backtest(data)
        final_balance = self.simulate_trades(data_with_recommendations, initial_balance)
        return accuracy, final_balance
