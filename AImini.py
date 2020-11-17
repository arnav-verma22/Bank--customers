# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv('Churn_Modelling.csv')
print(df.head(5))
y = df['Exited']
x = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])


