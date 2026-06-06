import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.model_selection import GridSearchCV
df= pd.read_csv(r"D:\ML\heart.csv")
df.drop_duplicates(inplace=True)
num_cols=['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak','ca']
cat_cols=['cp','restecg','slope','thal']
x=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y=df["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)
##numerical pipeline
num_pipeline= Pipeline([
    ("imputer",SimpleImputer(strategy="median")), ("scaler",StandardScaler())])
cat_pipeline= Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder())])
preprocessor= ColumnTransformer([
    ("cat",cat_pipeline,cat_cols),
    ("num",num_pipeline,num_cols)
])
pipe= Pipeline([
    ("preprocessor", preprocessor),
    ("model",KNeighborsClassifier())
])
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
params = {
    "model__n_neighbors": [3,5,7,9,11],
    "model__weights": ["uniform","distance"],
    "model__metric": ["euclidean","manhattan"]
}
grid = GridSearchCV(pipe, params, cv=5, scoring="f1")
grid.fit(x_train, y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("F1:",f1_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))