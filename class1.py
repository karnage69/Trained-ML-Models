import numpy as np      
import pandas as pd                  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score
#open the data 
data= r"D:\ML\Advertising.csv"
df= pd.read_csv(data)
#print top 5
print(df.head())
#data cleaning 
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
df[['TV', 'Radio', 'Newspaper', 'Sales']].boxplot()
plt.show()
#removing outliers 
def outlier(df,column):
    for col in column:
        q1= df[col].quantile (0.25)
        q3= df[col].quantile(0.75)
        IQR= q3-q1
        lower= q1- 1.5*IQR
        upper= q3+ 1.5*IQR
        df = df[(df[col] >= lower)&(df[col] <= upper)]
    return df
df = outlier(df, ['Newspaper'])
#feature engineering
df['TV_Radio'] = df['TV'] * df['Radio']
df['TV_sq'] = df['TV'] ** 2
df['Radio_sq'] = df['Radio'] ** 2
#define x and y 
X = df[['TV', 'Radio', 'Newspaper', 'TV_Radio', 'TV_sq', 'Radio_sq']]
y=df['Sales']
#train test split
x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
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