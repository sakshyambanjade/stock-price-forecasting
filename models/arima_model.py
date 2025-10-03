import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

def test_stationarity(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if dftest[1] <= 0.05:
        print("Series is stationary")
        return True
    else:
        print("Series is non-stationary")
        return False

def plot_series_acf_pacf(series, lags=40):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plot_acf(series, ax=plt.gca(), lags=lags)
    plt.subplot(122)
    plot_pacf(series, ax=plt.gca(), lags=lags)
    plt.show()

def find_best_arima_params(timeseries, p_range=3, d_range=2, q_range=3):
    import itertools
    p = range(0, p_range+1)
    d = range(0, d_range+1)
    q = range(0, q_range+1)
    pdq = list(itertools.product(p, d, q))
    best_aic = np.inf
    best_param = None
    for param in pdq:
        try:
            model = ARIMA(timeseries, order=param)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
        except:
            continue
    print(f"Best ARIMA params: {best_param} with AIC={best_aic}")
    return best_param

def fit_arima_model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def forecast_arima(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast

if __name__ == "__main__":
    import pathlib
    base_dir = pathlib.Path(__file__).parent.parent
    processed_file = base_dir / 'data' / 'processed' / 'AAPL_stock_data_clean.csv'

    df = pd.read_csv(processed_file, index_col='Date', parse_dates=True)
    close_series = df['Close']

    # Stationarity test
    is_stationary = test_stationarity(close_series)
    if not is_stationary:
        close_series = close_series.diff().dropna()
        print("After differencing:")
        test_stationarity(close_series)

    # Plot ACF and PACF for differenced series
    plot_series_acf_pacf(close_series)

    # Find optimal ARIMA params by AIC
    best_order = find_best_arima_params(close_series)

    # Fit ARIMA model
    model_fit = fit_arima_model(close_series, best_order)

    # Forecast next 30 days
    forecast = forecast_arima(model_fit, steps=30)
    print("Forecasted Values:", forecast)

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(close_series, label="Historical")
    plt.plot(forecast.index, forecast, label="Forecast", color="red")
    plt.title("ARIMA Model Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
