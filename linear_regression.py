import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
#import data
df=pd.read_csv(r"D:\ML\archive\student_scores.csv")
print(df.head())
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df["Hours"],
    y=df["Scores"]
)
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
#clean data
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
def outliers(df,columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR= q3 - q1
        lower= q1 - 1.5*IQR
        upperr= q3 - 1.5*IQR
        df[(df[col]>=lower)&(df[col]<=upperr)]
    return df
df= outliers(df,["Hours","Scores"])
#defien x and y
x= df[["Hours"]]
y= df["Scores"]
X_train, X_test, y_train, y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model= LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))