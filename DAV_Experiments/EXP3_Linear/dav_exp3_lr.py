
# ***Simple Linear Regression***


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

tesla_data = pd.read_csv("/content/TSLA.csv")

tesla_data.head(10)

tesla_data.info()

tesla_data.describe()

tesla_data.isnull().sum()

tesla_data.duplicated().sum()

tesla_data = tesla_data.dropna()

tesla_data.plot(kind='line', figsize=(8,6))

tesla_data['Date'].value_counts()

#converting int64 dtype to datetime64 dtype
tesla_data['Date']=pd.to_datetime(tesla_data['Date'])
tesla_data['Date']

tesla_data['High'].hist()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(tesla_data['Date'], tesla_data['Close'], label="Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

import seaborn as sns

features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
sns.pairplot(tesla_data[features])

plt.tight_layout()

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

fig, axs = plt.subplots(nrows=len(features), figsize=(8, 6 * len(features)))

for i, feature in enumerate(features):
    sns.kdeplot(data=tesla_data[feature], ax=axs[i], color='skyblue', fill=True)
    axs[i].set_title(f'KDE Plot of {feature}')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

X = tesla_data[['High','Low','Open','Volume']].values
Y = tesla_data['Close'].values

X

Y

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

y_pred

from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
print("R-squared Score:", r_squared)

# Evaluate Linear Regression
from sklearn.metrics import mean_absolute_error
# Evaluate the model on the test data.
score = lr_model.score(X_test, y_test)

# Print the score.
print('The score of the model is:', score)

lr_mae = mean_absolute_error(y_test, y_pred)
print(f'\nLinear Regression Mean Absolute Error: {lr_mae}')

data1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted' : y_pred.flatten()})

data1

import math
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test,y_pred)))

data1.sort_values(by='Actual', inplace=True)

plt.scatter(data1['Actual'], data1['Predicted'], color='blue', label='Actual vs Predicted')

x = np.linspace(data1['Actual'].min(), data1['Actual'].max(), 100)
plt.plot(x, x, color='red', label='Best Fit Line')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.legend()

plt.show()

graph = data1.head(50)
graph.plot(x='Actual', y='Predicted', kind='line')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()

graph = data1.head(100)
graph.plot(x='Actual', y='Predicted', kind='scatter')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (Scatter Plot)')
plt.show()

graph_data = data1.head(100)
plt.figure(figsize=(12,6))
sns.lmplot(x='Actual', y='Predicted', data=graph_data, line_kws={'color': 'red'})

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (Regression Plot)')
plt.tight_layout()
plt.show()