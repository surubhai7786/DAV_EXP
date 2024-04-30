import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1,3,10,16,26,36]).reshape((-1,1))
y = np.array([42,50,75,100,150,200])

model = LinearRegression()

model.fit(x,y)

print(f'Coefficient of regression: {model.coef_}')
print(f'Y-Intercept: {model.intercept_}')

y_pred = model.predict(x)

plt.scatter(x, y, color = 'lightcoral')
plt.plot(x, y_pred, color='teal')
plt.title('Linear Regression')
plt.box(False)
plt.show()