import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import cifar10
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Carregar o conjunto de dados CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Definir classes de interesse (Gato e Não Gato)
classes = ["Avião", "Automóvel", "Pássaro", "Gato", "Cervo", "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão"]

# Função para converter uma imagem em um vetor binário
def binarize_image(image):
    binary_image = np.where(image > 127, 1, 0)
    return binary_image.flatten()

# Aplicar a binarização às imagens de treinamento e teste
train_data = np.array([binarize_image(image) for image in train_images])
test_data = np.array([binarize_image(image) for image in test_images])

# Reduzir o tamanho do conjunto de dados (opcional para agilizar o processo de treinamento)
sample_size = 5000
train_data = train_data[:sample_size]
train_labels = train_labels[:sample_size]
test_data = test_data[:sample_size]
test_labels = test_labels[:sample_size]

# Inicializar o classificador k-NN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinar o classificador com os dados de treinamento
knn.fit(train_data.reshape(len(train_data), -1), train_labels)

# Fazer previsões com os dados de teste
predictions = knn.predict(test_data.reshape(len(test_data), -1))

# Avaliar a precisão do modelo
accuracy = np.mean(predictions == test_labels)
print(f"Precisão do modelo: {accuracy * 100:.2f}%")

# Plotar algumas imagens de teste com suas previsões
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"Real: {classes[test_labels[i][0]]}\nPrevisto: {classes[predictions[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
