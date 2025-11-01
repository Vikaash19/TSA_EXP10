# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 1.11.2025

## AIM:
To implement SARIMA model using python.
## ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('gold.csv')
data['Date'] = pd.to_datetime(data['Date'])

plt.plot(data['Date'], data['Close'])
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('Gold Price Time Series')
plt.show()

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['Close'])

plot_acf(data['Close'])
plt.show()
plot_pacf(data['Close'])
plt.show()


train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
## OUTPUT:
### Original dataset:
<img width="758" height="568" alt="1 original series" src="https://github.com/user-attachments/assets/4234134f-0a41-4ba1-ac71-43c956b7070a" />

### ACF:
<img width="755" height="545" alt="2 acf" src="https://github.com/user-attachments/assets/074f61e1-7c73-44f6-aba2-3a46a1095534" />

### PACF:
<img width="741" height="542" alt="3 pacf" src="https://github.com/user-attachments/assets/7e40b844-9ae9-4f4a-96b9-dfe31a41d9b7" />

### RMSE:
<img width="255" height="33" alt="4 rmse" src="https://github.com/user-attachments/assets/d8eca791-c6f3-4857-9a01-2321aad2edf2" />

### SARIMA Model:
<img width="747" height="563" alt="5 sarima model" src="https://github.com/user-attachments/assets/fc4f73da-f760-4044-955b-14273331ef50" />

## RESULT:
Thus the program run successfully based on the SARIMA model.
