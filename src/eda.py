import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    return df

def plot_time_series(df, column):
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df[column], label=f'{column} Price')
    plt.title(f'{column} Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trading_volume(df):
    plt.figure(figsize=(14,4))
    plt.bar(df.index, df['Volume'], color='skyblue')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.show()

def add_features(df):
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def plot_correlation_matrix(df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

def main():
    df = load_data('data/processed/AAPL_stock_data_clean.csv')
    print(df.head())

    plot_time_series(df, 'Close')
    plot_trading_volume(df)

    df = add_features(df)
    print(df[['MA7', 'MA30', 'Daily_Return']].head())

    plot_correlation_matrix(df)

if __name__ == "__main__":
    main()
