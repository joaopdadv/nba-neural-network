import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from modules.activation_functions import Sigmoidal, Softmax
from modules.structures import Layer, Network
from modules.loss_functions import MSELoss

# Carregar dados
data = pd.read_csv('players_data.csv')
data_copia = data.copy()

# Remover colunas Player, Performance, Pos
data = data.drop(columns=['Player', 'Performance', 'Pos', 'Tm'])
data_copia = data_copia.drop(columns=['Pos', 'Player', 'Tm'])

# transformar dados em float
data = data.replace(',', '.', regex=True).astype(float)

# transformar performance de data_copia em dummy usando pandas
data_copia = pd.get_dummies(data_copia, columns=['Performance'], dtype=float)

# Normalizar dados de todas colunas
data = (data - data.min()) / (data.max() - data.min())

print(data_copia.head(3))
print(data.head(3))
print(data.shape)


layer1 = Layer(data.shape[1], 10) # (27, 10)
sigmoidal1 = Sigmoidal()
layer2 = Layer(10, 2) # (10, 2)
s = Softmax()

network = Network([layer1, sigmoidal1, layer2, s])

# print(layer1.W) # (27, 10) - 27 vetores de 10 elementos
# print(layer1.b) # (10,)

# get first row of data
first_row = data.iloc[0]

# get first row wanted response from the data_copia performance (it was dummified to performance_Good and performance_Bad)
target = data_copia.iloc[0][['Performance_Good', 'Performance_Bad']]

# print(layer1.forward(first_row.to_list())) # (10,)
# print(layer2(sigmoidal1(layer1(first_row.to_list())))) # (10,)
# print(network(first_row.to_list())) # (10,)

prediction = network(first_row.to_list()); # (2,)

# calcular perda
print(prediction)
print([target.to_list()[i].item() for i in range(target.size)])

# erro na camada de saída
mse = MSELoss()
loss = mse(prediction, target.to_list())
print(loss)

# erro nas camadas ocultas
# pegar gradiente da camada de saída
grad = mse.backward()
print("Gradient:", grad)
# propagar para trás