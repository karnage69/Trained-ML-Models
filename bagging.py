import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.ensemble import BaggingClassifier
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
base_model = DecisionTreeClassifier()
model = BaggingClassifier(estimator=base_model,n_estimators=100,max_samples=0.8,bootstrap=True,random_state=42)
pipe= Pipeline([
    ("preprocessor", preprocessor),
    ("model",model)
])
# Hyperparameter tuning
params = {
    "model__n_estimators": [50, 100, 150],
    "model__max_samples": [0.6, 0.8, 1.0],
    "model__estimator__max_depth": [3, 5, 10]
}

grid = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
# Train
grid.fit(x_train, y_train)

best_model = grid.best_estimator_


# Predict
train_pred = best_model.predict(x_train)
test_pred = best_model.predict(x_test)


# Metrics
print("Best Params:", grid.best_params_)

print("\nTrain Accuracy:",
      accuracy_score(y_train, train_pred))

print("Test Accuracy:",
      accuracy_score(y_test, test_pred))

print("\nPrecision:",
      precision_score(y_test, test_pred))

print("Recall:",
      recall_score(y_test, test_pred))

print("F1 Score:",
      f1_score(y_test, test_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, test_pred))