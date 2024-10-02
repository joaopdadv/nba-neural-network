import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from modules.activation_functions import Sigmoidal
from modules.structures import Layer, Network

# Carregar dados
data = pd.read_csv('players_data.csv')
data_copia = data.copy()

# Remover colunas Player, Performance, Pos
data = data.drop(columns=['Player', 'Performance', 'Pos', 'Tm'])
data_copia = data_copia.drop(columns=['Pos', 'Player', 'Tm'])

# transformar dados em float
data = data.replace(',', '.', regex=True).astype(float)


# Normalizar dados de todas colunas
data = (data - data.min()) / (data.max() - data.min())

print(data.head(3))
print(data.shape)


layer1 = Layer(data.shape[1], 10)
sigmoidal1 = Sigmoidal()

# print(layer1.W) # (27, 10) - 27 vetores de 10 elementos
# print(layer1.b) # (10,)

# get first row of data
first_row = data.iloc[0]

print(layer1.forward(first_row.to_list())) # (10,)
print(sigmoidal1(layer1(first_row.to_list()))) # (10,)