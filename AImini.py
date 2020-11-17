# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv('Churn_Modelling.csv')
print(df.head(5))
y = df['Exited']
x = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])
print(x)

'''from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = ct.fit_transform(x)'''

dummies = pd.get_dummies(x['Geography'])
x = pd.concat([x, dummies], axis=1)
x.drop(columns=['Geography'], inplace=True)
print(x)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

