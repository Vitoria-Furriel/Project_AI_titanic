import pandas as pd
import numpy as np

dataset = pd.read_csv("Titanic.csv", sep=";" , encoding="utf-8-sig")

dataset = dataset.drop(["Name", "Ticket", "Cabin" , "PassengerId"], axis=1)
dataset.head()

X = dataset.drop("Survived", axis=1)
y = dataset["Survived"] 


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

categorial_features = ["Sex", "Embarked"]


ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),categorial_features)],remainder="passthrough")

X_transformed = ct.fit_transform(X)

