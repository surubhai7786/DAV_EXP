

# ***Data Analytics Libraries In Python***

***1) Pandas***
"""

import pandas as pd

data = pd.read_csv('/content/Employee.csv')
data.head(10)

# Check for missing values in the DataFrame
print("\nMissing values in the DataFrame:")
data.isnull().sum()

# Check for missing values in the DataFrame
print("\nDuplicated values in the DataFrame:")
data.duplicated().sum()

data = data.drop_duplicates()
data.duplicated().sum()

# Group by City and calculate the average Age and ExperienceInCurrentDomain
city_stats = data.groupby('City').agg({'Age': 'mean', 'ExperienceInCurrentDomain': 'mean'})
print("\nAverage Age and ExperienceInCurrentDomain by City:")
city_stats

data.info()

print("\nSummary statistics of numerical columns:")
data.describe()

"""***2) Numpy***"""

import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Calculate mean and standard deviation
print("\nMean of the array:", np.mean(arr))
print("\nStandard deviation of the array:", np.std(arr))

# Multiply all elements of the array by 2
print("\nArray multiplied by 2:")
print(arr * 2)

# Create a random array of shape (2, 3) with values between 0 and 1
random_array = np.random.rand(2, 3)
print("\nRandom Array:")
print(random_array)

# Element-wise addition
print("\nElement-wise addition with random array:")
print(arr + random_array)

# Element-wise multiplication
print("\nElement-wise multiplication with random array:")
print(arr * random_array)

# Dot product of two arrays
another_array = np.array([[2, 3], [4, 5], [6, 7]])
print("\nDot product with another array:")
print(np.dot(arr, another_array))

"""***3) Matplotlib***"""

import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.show()

#Scatter Plot
# Generate random data
x = np.random.rand(70)
y = np.random.rand(70)
colors = np.random.rand(70)
sizes = 100 * np.random.rand(70)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap='viridis')
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar(label='Color Intensity')
plt.show()

#Histogram
# Generate some random data
data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Random Data')
plt.show()

"""***4) Scikit-Learn***"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

emp_data = pd.read_csv('/content/Employee.csv')
emp_data.head()

non_numeric_columns = emp_data.select_dtypes(exclude=['number']).columns

# Apply one-hot encoding to non-numeric columns
emp_data_encoded = pd.get_dummies(emp_data, columns=non_numeric_columns)

X = emp_data_encoded.drop('LeaveOrNot', axis=1)
Y = emp_data_encoded['LeaveOrNot']

X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size=0.3, random_state=42)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, Y_train)

y_pred = lr_model.predict(X_test_scaled)
y_pred

mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Coefficients:", lr_model.coef_)

"""***Seaborn***"""

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your pandas DataFrame
# Example plot: Boxplot of Age grouped by Gender
sns.boxplot(x='Gender', y='Age', data=emp_data)
plt.title('Boxplot of Age grouped by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# Assuming df is your pandas DataFrame
# Example plot: Scatter plot of Age vs. ExperienceInCurrentDomain colored by Gender
sns.scatterplot(x='Age', y='ExperienceInCurrentDomain', hue='Gender', data=emp_data)
plt.title('Scatter plot of Age vs. ExperienceInCurrentDomain')
plt.xlabel('Age')
plt.ylabel('ExperienceInCurrentDomain')
plt.show()