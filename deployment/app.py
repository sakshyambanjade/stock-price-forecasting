import gradio as gr
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import io
from PIL import Image
import sys

# Add project root to sys.path for imports
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from models.arima_model import fit_arima_model, forecast_arima
from tensorflow.keras.models import load_model
from models.prophet_model import train_prophet_model, make_forecast

# Load data
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_FILE = BASE_DIR / 'data' / 'processed' / 'AAPL_stock_data_clean.csv'
df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)

# Correctly handle tz-aware datetime index:
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index, utc=True)  # Convert with utc=True to fix tz-aware
else:
    # If already DatetimeIndex, ensure tz-naive
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img

def arima_forecast(days: int):
    close_series = df['Close']

    # Ensure freq set for ARIMA
    if close_series.index.freq is None:
        freq = pd.infer_freq(close_series.index)
        close_series = close_series.asfreq(freq)

    model_fit = fit_arima_model(close_series, order=(5,1,0))
    forecast_obj = model_fit.get_forecast(steps=days)
    last_date = close_series.index[-1]
    freq = close_series.index.freqstr if close_series.index.freqstr else pd.infer_freq(close_series.index)
    forecast_index = pd.date_range(start=last_date, periods=days+1, freq=freq)[1:]
    forecast = forecast_obj.predicted_mean
    forecast.index = forecast_index
    plt.figure(figsize=(10,5))
    plt.plot(close_series.index, close_series, label='Historical')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title('ARIMA Forecast')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return plot_to_image(fig)

def lstm_forecast(days: int):
    from sklearn.preprocessing import MinMaxScaler
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    try:
        model = load_model(str(BASE_DIR / 'models' / 'lstm_model.h5'))
    except Exception:
        return 'LSTM model file not found or corrupted. Please train and save lstm_model.h5 first.', None
    last_60_days = scaled_data[-60:]
    pred_list = []
    current_batch = last_60_days.reshape((1, 60, 1))
    for _ in range(days):
        pred = model.predict(current_batch)[0]
        pred_list.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    pred_list = scaler.inverse_transform(pred_list)
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Close'], label='Historical')
    plt.plot(forecast_dates, pred_list.flatten(), label='Forecast')
    plt.title('LSTM Forecast')
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    return plot_to_image(fig)

def prophet_forecast(days: int):
    df_prophet = df.reset_index()
    df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_localize(None)
    split_date = pd.Timestamp(df_prophet['ds'].max())
    df_train = df_prophet[df_prophet['ds'] < split_date]
    model = train_prophet_model(df_train)
    forecast = make_forecast(model, periods=int(days))
    plt.figure(figsize=(10,5))
    model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.tight_layout()
    fig = plt.gcf()
    return plot_to_image(fig)

def predict(model_name, days):
    if days <= 0:
        return 'Please select a positive number of days.', None
    if model_name == 'ARIMA':
        img = arima_forecast(days)
    elif model_name == 'LSTM':
        img = lstm_forecast(days)
        if isinstance(img, tuple):
            return img
    elif model_name == 'Prophet':
        img = prophet_forecast(days)
    else:
        return 'Unknown model selected.', None
    return '', img

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=['ARIMA', 'LSTM', 'Prophet'], label='Select Model'),
        gr.Slider(minimum=1, maximum=60, value=10, label='Forecast Days')
    ],
    outputs=[
        gr.Textbox(label='Message'),
        gr.Image(type='pil', label='Forecast Plot')
    ],
    title='Stock Price Forecasting',
    description='Select a forecasting model and forecast days for Apple stock prices.'
)

if __name__ == '__main__':
    iface.launch()
