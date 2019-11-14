import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()


wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)


columns_names = wine.feature_names
print(columns_names)
print(wine.target)
y = wine.target
X = wine.data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)


print(f"Intercept per class: {lr.intercept_}\n")
print(f"Coeficients per class: {lr.coef_}\n")

print(f"Available classes: {lr.classes_}\n")
print(f"Named Coeficients for class 1: {pd.DataFrame(lr.coef_[0], columns_names)}\n")
print(f"Named Coeficients for class 2: {pd.DataFrame(lr.coef_[1], columns_names)}\n")
print(f"Named Coeficients for class 3: {pd.DataFrame(lr.coef_[2], columns_names)}\n")

print(f"Number of iterations generating model: {lr.n_iter_}")

from sklearn import neighbors, datasets


knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

print(knn.score(X_test, y=y_test))