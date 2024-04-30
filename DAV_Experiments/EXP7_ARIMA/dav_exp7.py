

# ***ARIMA MODEL***


#Import Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

#Loading Dataset
air_ps_dataset = pd.read_csv('/content/AirPassengers.csv')
air_ps_dataset.head(10)

"""***Data Analysis***"""

air_ps_dataset.shape

air_ps_dataset.isnull().sum()

air_ps_dataset.duplicated().sum()

air_ps_dataset.info()

air_ps_dataset.describe()

air_ps_dataset['Month'] = pd.to_datetime(air_ps_dataset['Month'])

air_ps_indexedData = air_ps_dataset.set_index('Month')
air_ps_indexedData.head()

plt.plot(air_ps_indexedData, color='blue')
plt.xlabel('Date')
plt.ylabel('Number of passengers')
plt.show()

"""***4 Month Moving Average***"""

#Rolling Mean
four_months_moving_average = air_ps_indexedData.rolling(window=4).mean()

plt.plot(air_ps_indexedData, color='blue', label='Original')
plt.plot(four_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('4 Months Moving Average')
plt.show()

"""***6-Months Moving Average***"""

six_months_moving_average = air_ps_indexedData.rolling(window=6).mean()

plt.plot(air_ps_indexedData, color='green', label='Original')
plt.plot(six_months_moving_average, color='blue', label='Rolling Mean')
plt.legend(loc='best')
plt.title('6 Months Moving Average')
plt.show()

"""***8-Month Moving Average***"""

eight_months_moving_average = air_ps_indexedData.rolling(window=8).mean()

plt.plot(air_ps_indexedData, color='blue', label='Original')
plt.plot(eight_months_moving_average, color='orange', label='Rolling Mean')
plt.legend(loc='best')
plt.title('8 Months Moving Average')
plt.show()

"""***12-Month Moving Average***"""

twelve_months_moving_average = air_ps_indexedData.rolling(window=12).mean()

plt.plot(air_ps_indexedData, color='green', label='Original')
plt.plot(twelve_months_moving_average, color='red', label='Rolling Mean')
plt.legend(loc='best')
plt.title('12 Months Moving Average')
plt.show()

"""***Rolling statistics***"""

# Rolling Mean and Std Dev
rollmean = air_ps_indexedData.rolling(window=12).mean()
print("\nRolling Mean :\n",rollmean)

rollstd = air_ps_indexedData.rolling(window=12).std()
print("\nRolling Standard Deviation :\n",rollstd)

plt.plot(air_ps_indexedData, color='blue', label='Original')
plt.plot(rollmean, color='red', label='Rolling Mean')
plt.plot(rollstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard deviation\n')
plt.show()

"""***Dickey Fuller test***"""

print("Result of Dicky-Fuller Test:\n")
dftest = adfuller(air_ps_indexedData['#Passengers'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
for key,value in dftest[4].items():
  dfoutput['Critical Value (%s)' %key] = value

dfoutput

"""***Estimating trend***"""

indexedData_logScale= np.log(air_ps_indexedData)

plt.plot(indexedData_logScale, 'blue')
plt.show()

movingAverage = indexedData_logScale.rolling(window=12).mean()
movingSTD = indexedData_logScale.rolling(window=12).std()
plt.plot(indexedData_logScale, color='blue')
plt.plot(movingAverage, color='red')

dataLogScaleMinusMovingAverage = indexedData_logScale - movingAverage
dataLogScaleMinusMovingAverage.dropna(inplace=True)
dataLogScaleMinusMovingAverage.head()

def test_stationarity(timeseries):

    # Rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation\n')
    plt.show(block=False)

    # Dickey Fuller test
    print('\n\nResults of Dickey Fuller Test:\n')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'No. of Lags used', 'Number of observations used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)

test_stationarity(dataLogScaleMinusMovingAverage)

"""***Another transformation***"""

exponentialDecayWeightedAverage = indexedData_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedData_logScale, color='blue', label='Original Data')
plt.plot(exponentialDecayWeightedAverage, color='red', label='Exponential Decay Weighted Average')
plt.legend()
plt.show()

dataLogScaleMinusMovingExponentialDecayAverage = indexedData_logScale - exponentialDecayWeightedAverage
test_stationarity(dataLogScaleMinusMovingExponentialDecayAverage)

"""***Another way of making the time series stationary is by differencing.***"""

dataLogDiffShifting = indexedData_logScale - indexedData_logScale.shift()
plt.plot(dataLogDiffShifting, color='blue')

dataLogDiffShifting.dropna(inplace=True)

test_stationarity(dataLogDiffShifting)

"""***Components of time series :- Seasonal decompose***"""

decomposition = seasonal_decompose(indexedData_logScale)

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 8))

# Plot original time series
axes[0].plot(indexedData_logScale, label='Original')
axes[0].set_ylabel('Original')

# Plot trend
axes[1].plot(trend, label='Trend')
axes[1].set_ylabel('Trend')

# Plot seasonal component
axes[2].plot(seasonal, label='Seasonal')
axes[2].set_ylabel('Seasonal')

# Plot residual
axes[3].plot(residual, label='Residual')
axes[3].set_ylabel('Residual')

# Add titles and legend
plt.suptitle('Decomposition of multiplicative time series')
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData )

"""***ACF and PACF plots***"""

fig, axes = plt.subplots(2, 1, sharey=False, sharex=False, figsize=(12, 8))

plot_acf(dataLogDiffShifting, lags=20, ax=axes[0], alpha=0.5)
plot_pacf(dataLogDiffShifting, lags=20, ax=axes[1], alpha=0.5)

plt.tight_layout()
plt.show()

"""***ARIMA Model***"""

# ARIMA model
model = ARIMA(indexedData_logScale, order=(2, 1, 2))
arima_model = model.fit()

arima_model.summary()

# Plot residual errors
residuals = pd.DataFrame(arima_model.resid)
residuals.plot()
plt.title('Residuals')
plt.show()

# Calculate residuals
residuals = arima_model.fittedvalues - dataLogDiffShifting['#Passengers']

# Calculate RSS
rss = sum(residuals**2)

# Plot RSS
plt.plot(residuals, color='blue')
plt.title('Residuals of ARIMA Model (RSS: %.4f)' % rss)
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

predictions_ARIMA_diff = pd.Series(arima_model.fittedvalues, copy=True)

if predictions_ARIMA_diff.iloc[0] == 0:
    # If the first value is zero, replace it with the first value of the original data
    original_first_value = dataLogDiffShifting.iloc[0]  # You need to replace 'original_data' with your actual original data
    predictions_ARIMA_diff.iloc[0] = original_first_value

predictions_ARIMA_diff.head()

plt.plot(predictions_ARIMA_diff)
plt.show()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(indexedData_logScale['#Passengers'].iloc[0], index=indexedData_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

# Predicting using the ARIMA model
predicted_values = arima_model.predict(typ='levels')
predictions_ARIMA_log = pd.Series(predicted_values, copy=True)
predictions_ARIMA = np.exp(predictions_ARIMA_log)

# Plot actual vs predicted values
plt.plot(air_ps_indexedData, color='blue', label='Actual')
plt.plot(predictions_ARIMA, color='orange', label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Date')
plt.ylabel('Log of Number of Passengers')
plt.show()