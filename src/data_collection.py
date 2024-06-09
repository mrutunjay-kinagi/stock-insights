import yfinance as yf

class DataCollector:
    def __init__(self, ticker_list):
        self.ticker_list = ticker_list

    def fetch_data(self, start_date, end_date):
        data = {}
        for ticker in self.ticker_list:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            data[ticker] = stock_data
        return data

    def save_data(self, data, directory):
        for ticker, df in data.items():
            file_path = f"{directory}/{ticker}.csv"
            df.to_csv(file_path)

if __name__ == "__main__":
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    collector = DataCollector(tickers)
    data = collector.fetch_data('2020-01-01', '2023-01-01')
    collector.save_data(data, '../data/raw')
