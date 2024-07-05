import yfinance as yf
import pandas as pd

class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_data(self):
        stock = yf.Ticker(self.ticker)
        fundamental_data = self.get_fundamental_data(stock)
        technical_data = self.get_technical_data(stock)
        return fundamental_data, technical_data

    def get_fundamental_data(self, stock):
        info = stock.info
        keys = ['shortName', 'sector', 'industry', 'marketCap', 'dividendYield', 'priceToBook', 'trailingPE', 'forwardPE', 'pegRatio', 'earningsGrowth', 'revenueGrowth', 'debtToEquity']
        fundamental_data = {key: info.get(key) for key in keys}
        fundamental_df = pd.DataFrame([fundamental_data])
        return fundamental_df

    def get_technical_data(self, stock):
        hist = stock.history(period="5y")
        hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['Signal Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['RSI'] = self.calculate_rsi(hist['Close'])
        return hist

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
