# Importing the necessary libraries
import pandas as pd
import numpy as np

dataset = pd.read_csv("Titanic.csv", sep=";" , encoding="utf-8-sig") #read the dataset

dataset = dataset.drop(["Name", "Ticket", "Cabin" , "PassengerId"], axis=1)
dataset.head()

X = dataset.drop("Survived", axis=1) #já cria o dataframe desejado, todos sem o survived TREINAMENTO
y = dataset["Survived"] #Colocamos direto para não errarmos o idex/posição. TESTE
X.head()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

categorial_features = ["Sex", "Embarked"] #Definimos quais são os features a serem transformados.

"""# Tranformando"""

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),categorial_features)],remainder="passthrough")

X_transformed = ct.fit_transform(X)
#temos que colocar X porque é nele que está Sex e Embarked para transformar.

#Foram transformadas com sucesso.
#As colunas transformadas, o OHE colocam no inicio da dataframe.
#Enquanto remainder ="passthrough" são colocados depois.

#Pelo fato de ter colocado dataset dentro do transoform

print(X_transformed[:5])

