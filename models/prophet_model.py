import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], utc=True)
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

def train_prophet_model(df_train):
    model = Prophet()
    model.fit(df_train)
    return model

def make_forecast(model, periods, freq='D'):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def evaluate_forecast(df_test, forecast):
    df_test = df_test.set_index('ds')
    forecast = forecast.set_index('ds')
    y_true = df_test['y']
    y_pred = forecast['yhat'].loc[df_test.index]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title('Stock Price Forecast with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    import pathlib
    base_dir = pathlib.Path(__file__).parent.parent
    processed_file = base_dir / 'data' / 'processed' / 'AAPL_stock_data_clean.csv'

    df = load_and_prepare_data(processed_file)

    max_date = df['ds'].max()
    min_date = max_date - timedelta(days=365*3)  # last 3 years of data
    df_train = df[(df['ds'] >= min_date) & (df['ds'] < max_date)]
    df_test = df[df['ds'] >= max_date]

    model = train_prophet_model(df_train)

    forecast = make_forecast(model, periods=30)  # forecast next 30 days

    evaluate_forecast(df_test, forecast)
    plot_forecast(model, forecast)
