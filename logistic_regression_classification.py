import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import (accuracy_score,
precision_score,recall_score,
f1_score,confusion_matrix
)
df= pd.read_csv(r"D:\ML\heart.csv")
#plt.figure(figsize=(12,8))
#sns.heatmap(
#df.corr(numeric_only=True),
#annot=True,
#cmap="coolwarm"
#)
#plt.show()
df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
num_cols=[
'age',
'sex',
'trestbps',
'chol',
'fbs',
'thalach',
'exang',
'oldpeak',
'ca'
]
#for col in cols:
#    sns.boxplot(
#        x=df[col]
#    )
#   plt.show()
cat_cols=['cp','restecg','slope','thal']
#defining x and y
x=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3,stratify=y)
#pipeline for numerical data
numerical_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("standardscaler",StandardScaler())
])
cat_pipeline= Pipeline([
    ("imputers",SimpleImputer(strategy="most_frequent")),
    ("onehotencoder", OneHotEncoder())
])
preprocessor=ColumnTransformer([
("num",numerical_pipeline,num_cols),
("cat",cat_pipeline,cat_cols)
])
pipe=Pipeline([
("prep",preprocessor),
("model",LogisticRegression())
])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
sns.heatmap(
confusion_matrix(y_test,y_pred),
annot=True
)