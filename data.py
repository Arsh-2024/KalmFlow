import pandas as pd
import yfinance as yf

def load_data(config):
    if config['data']['source'] == 'yfinance':
        tickers = config['data']['tickers']
        start = config['data']['start']
        end = config['data']['end']
        df = yf.download(tickers, start=start, end=end)['Close']
        return df.ffill()
    elif config['data']['source'] == 'csv':
        path = config['data']['csv_path']
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    else:
        raise ValueError("Unknown data source")

def compute_log_returns(prices):
    return (prices / prices.shift(1)).apply(lambda x: pd.Series(np.log(x))).dropna()
