import random

class Layer():
    def __init__(self, din, dout) -> None:
        # inicializar vetor de pesos e bias aleatórios sem numpy
        self.W = [[random.random() for _ in range(din)] for j in range(dout)] # é uma matriz de dimensão din x dout
        self.b = [random.random() for _ in range(dout)] # é um vetor de dimensão dout

    def forward(self, x):  # x é o vetor de entrada
        # Multiplica cada valor de x pelos pesos correspondentes em W e soma o bias
        output = []
        for j in range(len(self.b)):  # Para cada neurônio na camada
            soma = 0
            for i in range(len(x)):  # Para cada entrada
                soma += x[i] * self.W[j][i]  # Soma ponderada
            soma += self.b[j]  # Adiciona o bias
            output.append(soma)  # Adiciona o resultado ao vetor de saída
        return output  # Vetor de saída de tamanho dout (vetor de somatórias)
    
    def backward():
        pass

    def __call__(self, x):
        return self.forward(x)

class Network():
    def __init__(self, blocks:list) -> None:
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self):
        pass

    def __call__(self, x):
        return self.forward(x)
    