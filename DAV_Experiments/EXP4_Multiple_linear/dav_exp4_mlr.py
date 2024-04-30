

# ***Multiple Linear Regression***


# Import Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

emission_data = pd.read_csv("/content/CO2Emission_data.csv")

emission_data.head(10)

"""***Data Analysis***"""

emission_data.shape

emission_data.isnull().sum()

emission_data.duplicated().sum()

emission_data.dropna()

emission_data.describe()

emission_data.info()

"""***Data Visualizaton***"""

car_counts = emission_data['Car'].value_counts()
# Plotting the counts
plt.figure(figsize=(8, 6))
car_counts.plot(kind='bar')
plt.xlabel('Car')
plt.ylabel('Count')
plt.title('Counts of Cars')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(emission_data['CO2'], bins=10, edgecolor='black')
plt.xlabel('CO2')
plt.ylabel('Frequency')
plt.title('Histogram of CO2 Emissions')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Volume vs CO2
plt.subplot(2, 2, 1)
plt.scatter(emission_data['Volume'], emission_data['CO2'])
plt.xlabel('Volume')
plt.ylabel('CO2')
plt.title('Volume vs CO2')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
# Weight vs CO2
plt.subplot(2, 2, 2)
plt.scatter(emission_data['Weight'], emission_data['CO2'])
plt.xlabel('Weight')
plt.ylabel('CO2')
plt.title('Weight vs CO2')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
# Volume vs Weight
plt.subplot(2, 2, 3)
plt.scatter(emission_data['Volume'], emission_data['Weight'])
plt.xlabel('Volume')
plt.ylabel('Weight')
plt.title('Volume vs Weight')
plt.tight_layout()
plt.show()

#Volume
plt.figure(figsize=(8, 6))
sns.boxplot(x=emission_data['Volume'])
plt.title('Boxplot of Volume')
plt.xlabel('Volume')
plt.tight_layout()
plt.show()

#Weight
plt.figure(figsize=(8, 6))
sns.boxplot(x=emission_data['Weight'])
plt.title('Boxplot of Weight')
plt.xlabel('Weight')
plt.tight_layout()
plt.show()

#CO2
plt.figure(figsize=(8, 6))
sns.boxplot(x=emission_data['CO2'])
plt.title('Boxplot of CO2')
plt.xlabel('CO2')
plt.tight_layout()
plt.show()

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.distplot(emission_data['CO2'])
plt.title("Distribution of target variable(CO2)")
plt.show()

#Relationship of CO2 with other features
sns.pairplot(emission_data, x_vars=['Weight', 'Volume'], y_vars='CO2', height=4, aspect=1, kind='scatter')
plt.title("#Relationship of CO2 with other features")
plt.show()

correlation = emission_data.corr()
correlation

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot = True, cmap = 'coolwarm')
plt.title("Correlation Between Features")
plt.tight_layout()
plt.show()

"""***Model Building***"""

X = emission_data[['Weight', 'Volume']].values
X

Y = emission_data['CO2'].values
Y

# Splitting Dataset
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)

mlr_model = LinearRegression()

mlr_model.fit(X_train , Y_train)

y_pred = mlr_model.predict(X_test)
y_pred

"""***Evaluate Model***"""

# Evaluate the model
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))
print('\nRoot Mean Square Error:', r2)
# Print the coefficients
print("\nCoefficients:", mlr_model.coef_)
print("\nIntercept:", mlr_model.intercept_)

#Printing the model coefficients
print('Intercept: ',mlr_model.intercept_)
# pair the feature names with the coefficients
print("\n")
list(zip(X, mlr_model.coef_))

# Visualize the predictions using a regression plot
plt.figure(figsize=(8, 6))
sns.regplot(x=Y_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Regression Plot of Actual vs. Predicted CO2 Emissions')
plt.grid(True)
plt.show()

model_diff = pd.DataFrame({'Actual value': Y_test.flatten() , 'Predicted value': y_pred.flatten()})
model_diff

plt.figure(figsize=(10, 6))
plt.plot(model_diff['Actual value'], label='Actual CO2 Emissions', marker='o')
plt.plot(model_diff['Predicted value'], label='Predicted CO2 Emissions', marker='x')
plt.xlabel('Index')
plt.ylabel('CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions (Line Plot)')
plt.legend()
plt.grid(True)
plt.show()

"""***Regression Plot for actual and predicted value***


"""

plt.figure(figsize=(10, 6))
sns.regplot(x='Actual value', y='Predicted value', data=model_diff, scatter_kws={"color": "blue"}, line_kws={"color": "green"}, label='Regression Line')
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('Predicted CO2 Emissions')
plt.title('Actual vs Predicted CO2 Emissions (Regression Plot)')
plt.legend()
plt.grid(True)
plt.show()

"""***END***"""