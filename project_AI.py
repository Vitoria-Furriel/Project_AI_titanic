# -*- coding: utf-8 -*-

# Importing the necessary libraries
import pandas as pd
import numpy as np

dataset = pd.read_csv("Titanic_1.csv", delimiter= ";") #read the dataset

X = dataset.drop("Survived", axis=1) #já cria o dataframe desejado, todos sem o survived TREINAMENTO
y = dataset["Survived"] #Colocamos direto para não errarmos o idex/posição. TESTE
X = dataset.drop(["Name", "Ticket", "Cabin" ], axis=1)
X.head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

print(dataset)

categorial_features = ["Sex", "Embarked"] #Definimos quais são os features a serem transformados.

"""# Tranformando"""

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),categorial_features)],remainder="passthrough")

X_transformed = ct.fit_transform(X)


print(X_transformed[:70])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state =42)

cols = [5,8,11]



sc = StandardScaler()
X_train[:,cols] = sc.fit_transform(X_train[:,cols])
X_test[:,cols] = sc.transform(X_test[:,cols])

print(X_train)