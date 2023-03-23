import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("company_revenue.csv")
print("Data Visualization")
print(df.head())

X_data = df[["R&D Spend", "Marketing Spend"]]
Y_data = df["Profit"]

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size =0.3)

model = LinearRegression()
model.fit(X_train, Y_train)

print("Data Used for Prediction")
print(X_test.head(5))

print("Predicted Value")
print(model.predict(X_test)[1:6])

print("Actual Value")
print(Y_test.head(5))

print("Model Score: ")
print(model.score(X_test,Y_test))