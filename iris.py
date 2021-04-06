import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Get Iris dataset from UCI archives and store locally
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Create Iris dataframe from CSV
iris_df = pd.read_csv(iris, sep=',')
headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_df.columns = headers

print(iris_df.head())

# Create X and y data
X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, 4].values

# Create train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Create StandardScaler object
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create KNeighborsClassifier object
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

# Evaluate the model using confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))