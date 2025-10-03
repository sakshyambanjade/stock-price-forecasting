import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

def save_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)
    print(f"Saved data to {filename}")

if __name__ == "__main__":
    df = download_stock_data('AAPL', '5y')
    print(df.head())
    save_csv(df, 'data/raw/AAPL_stock_data.csv')
