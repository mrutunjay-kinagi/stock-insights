import pandas as pd
import yfinance as yf

class DataCollector:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_fundamental_data(self, stock):
        info = stock.info
        data = {
            'Ticker': stock.ticker,
            'Short Name': info.get('shortName', None),
            'Sector': info.get('sector', None),
            'Industry': info.get('industry', None),
            'Country': info.get('country', None),
            'Market Cap': info.get('marketCap', None),
            'PE Ratio': info.get('trailingPE', None),
            'PB Ratio': info.get('priceToBook', None),
            'PS Ratio': info.get('priceToSalesTrailing12Months', None),
            'Profit Margin': info.get('profitMargins', None),
            'Operating Margin': info.get('operatingMargins', None),
            'ROE': info.get('returnOnEquity', None),
            'Earnings Growth': info.get('earningsGrowth', None),
            'Revenue Growth': info.get('revenueGrowth', None),
        }
        return data

    def fetch_technical_data(self, stock):
        hist = stock.history(period="5y")
        
        # Calculate technical indicators
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['Signal Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['RSI'] = self.compute_rsi(hist['Close'])
        
        return hist

    def compute_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_data(self):
        stock = yf.Ticker(self.ticker)
        
        # Fetch fundamental data
        fundamental_data = self.fetch_fundamental_data(stock)
        fundamental_df = pd.DataFrame(fundamental_data, index=[0])
        
        # Fetch technical data
        technical_data = self.fetch_technical_data(stock)
        
        return fundamental_df, technical_data

if __name__ == "__main__":
    collector = DataCollector('IDFCFIRSTB.NS')
    fundamental_data, technical_data = collector.get_data()
    
    # Display the fetched data
    print(f'Fundamental data for IDFCFIRSTB.NS:')
    print(fundamental_data)
    
    print(f'\nTechnical data for IDFCFIRSTB.NS:')
    print(technical_data.tail())
    
    # Export to CSV
    fundamental_data.to_csv('IDFCFIRSTB_NS_fundamental_data.csv', index=False)
    technical_data.to_csv('IDFCFIRSTB_NS_technical_data.csv')
    
    print('\nData exported to CSV files.')
