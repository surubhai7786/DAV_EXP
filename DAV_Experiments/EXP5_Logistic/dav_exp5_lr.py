
# ***Logistic Regression***

# Import Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Loading Dataset
wine_data = pd.read_csv("/content/winequality-red.csv")
wine_data.head(10)

"""***Exploratory Data Analysis***"""

wine_data.shape

# Checking Missing Values
wine_data.isnull().sum()

# Check duplicate values
wine_data.duplicated().sum()

wine_dataset = wine_data.drop_duplicates()

wine_dataset.duplicated().sum()

wine_dataset.describe()

wine_dataset.info()

"""***Data Visualization***"""

plt.figure(figsize=(8, 6))
sns.catplot(x='quality' , data = wine_dataset , kind = 'count',palette='Set2')
plt.show()

quality_counts = wine_dataset['quality'].value_counts()
# Plotting the counts
plt.figure(figsize=(8, 6))
quality_counts.plot(kind='bar')
plt.xlabel('Count')
plt.ylabel('Quality')
plt.title('Counts of quality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.displot(wine_dataset['fixed acidity'])
plt.show()

# volatile acidity vs Quality
plot = plt.figure(figsize=(8,6))
sns.barplot(x = 'quality' , y = 'volatile acidity' , data = wine_dataset , palette='husl')
plt.show()

# citric acid vs Quality
plot = plt.figure(figsize=(8,6))
sns.barplot(x = 'quality' , y = 'citric acid' , data = wine_dataset , palette='mako')
plt.show()

# chlorides vs Quality
plot = plt.figure(figsize=(8,6))
sns.barplot(x = 'quality' , y = 'chlorides' , data = wine_dataset , palette = "colorblind")
plt.show

plt.figure(figsize=(10, 6))
plt.scatter(wine_dataset['citric acid'], wine_dataset['quality'], alpha=0.5)
plt.title('Quality vs Citric Acid')
plt.xlabel('Citric Acid')
plt.ylabel('Quality')
plt.grid(True)
plt.show()

"""***Correlation***"""

correlation = wine_dataset.corr()

#Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation , cbar = True , square = True , fmt = '.1f' , annot = True , annot_kws = {'size':8} , cmap = 'Blues')
plt.show()

"""***Data Preprocessing***"""

# Separate the data and label
X = wine_dataset.drop('quality' , axis = 1)
X

"""***Label Binarization***"""

Y = wine_dataset['quality'].apply(lambda y_values : 1 if y_values>=7 else 0)

Y

"""***Splitting Data Into Train And Test***"""

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=3)

print(Y.shape , Y_train.shape , Y_test.shape)

"""***Training Model***"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lr_model = LogisticRegression(solver = "liblinear")
lr_model.fit(X_train_scaled, Y_train)

lr_model.feature_names_in_ = list(X_train.columns)
Y_pred = lr_model.predict(X_test)
Y_pred

lr_test_acc = accuracy_score(Y_test,Y_pred)

print("Accuracy Score of Logistic Regression Model:",lr_test_acc)

from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix

# Precision
precision = precision_score(Y_test, Y_pred)
print("\nPrecision:", precision)

# Recall
recall = recall_score(Y_test, Y_pred)
print("\nRecall:", recall)

# F1 Score
f1 = f1_score(Y_test, Y_pred)
print("\nF1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:\n")
conf_matrix

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

from sklearn.metrics import precision_recall_curve, auc
X_test_scaled = scaler.fit_transform(X_test)
Y_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(Y_test, Y_probs)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

from sklearn.metrics import roc_curve, auc
X_test_scaled = scaler.transform(X_test)
probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(Y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

"""***Input Data For Predictive System***"""

# Select a random row from the test set
random_index = np.random.randint(0, len(X_test))
print(random_index)
input_data = X_test_scaled[random_index].reshape(1, -1)

selected_row = X_test.iloc[random_index]
print("\nRandomly selected row from the test set:")
selected_row

"""***Logistic Regression Predictive Model***"""

threshold = 0.5
lr_model_probabilities = lr_model.predict_proba(input_data)
print(lr_model_probabilities)

if lr_model_probabilities[0][1] >= threshold:
    print("\nThe Quality Of Wine is Good !!!\n")
    print("Good Quality Wine! Savor your beverage and enjoy the moment!\n")
else:
    print("\nThe Quality Of Wine is Bad !!!\n")
    print("Bad quality, Avoid drinking!\n")