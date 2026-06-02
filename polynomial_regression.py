from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
#importing data
df= pd.read_csv("mpg.csv")
print(df.head())
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.scatter(df["weight"],df["mpg"])
plt.xlabel("weight")
plt.ylabel("mpg")
plt.title("mpg vs weight")
plt.subplot(2,2,2)
plt.scatter(df["horsepower"],df["mpg"])
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.title("horse power bs mpg")
plt.subplot(2,2,3)
plt.scatter(df["displacement"],df["mpg"])
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("displacement bs mpg")
plt.tight_layout()
plt.show()
#data cleaning 
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
#removing outliers
def outliers(df,columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR= q3 - q1
        lower= q1 - 1.5*IQR
        upperr= q3 - 1.5*IQR
        df[(df[col]>=lower)&(df[col]<=upperr)]
    return df
df = outliers(df,["horsepower", "weight", "displacement", "acceleration"])
#adding new features
df["cylinder_displacement"]=df["cylinders"]*df["displacement"]
df["weight_horsepower"]=df["weight"]*df["horsepower"]
df["displacement_squared"]=df['displacement']**2
plt.figure(figsize=(14,10))
# ORIGINAL
plt.subplot(2,2,1)
plt.scatter(df["weight"], df["mpg"])
plt.title("Weight vs MPG")
# NEW FEATURE
plt.subplot(2,2,2)
plt.scatter(df["weight_horsepower"], df["mpg"])
plt.title("Weight * Horsepower vs MPG")
# ORIGINAL
plt.subplot(2,2,3)
plt.scatter(df["displacement"], df["mpg"])
plt.title("Displacement vs MPG")
# NEW FEATURE
plt.subplot(2,2,4)
plt.scatter(df["cylinder_displacement"], df["mpg"])
plt.title("Cylinders * Displacement vs MPG")
plt.tight_layout()
plt.show()
#define x and y 
X = df[['cylinders','displacement','horsepower','weight',
        'acceleration','model_year',"cylinder_displacement",
        "weight_horsepower","displacement_squared"]]
y=df['mpg']
#train test split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
#polynomial regression feature
poly=  PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
#feature scaling
scaler= StandardScaler()
x_train_scale= scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)
#train model
model= LinearRegression()
model.fit(x_train_scale,y_train)
#predict
y_pred = model.predict(x_test_scale)
#evaluate
print(f"R^2  : {r2_score(y_test, y_pred):.4f}")
print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")
#actual vs prediction
plt.figure(figsize=(8, 5))

# Scatter plot: actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', label='Predictions')

# Perfect prediction line (y = x diagonal)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linewidth=2, label='Perfect Prediction'
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Polynomial Regression (degree=2): Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("poly_regression_plot.png", dpi=150)  # saves the plot
plt.show()
print("\nPlot saved as poly_regression_plot.png")
