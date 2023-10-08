# Classificação de Imagens de Gatos e Não Gatos com k-NN

Este projeto implementa um sistema de classificação de imagens de Gatos e Não Gatos utilizando o algoritmo k-Nearest Neighbors (k-NN) a partir de dados binários. O conjunto de dados utilizado é o CIFAR-10.

## Requisitos

- Python 3.x
- Bibliotecas Python: numpy, matplotlib, scikit-learn (sklearn), tensorflow

Você pode instalar as bibliotecas Python necessárias utilizando o pip:

```
pip install numpy matplotlib scikit-learn tensorflow
```

## Como Usar

1. Clone o repositório ou faça o download dos arquivos.

2. Execute o script `main.py`:

```
python main.py
```

O script carregará o conjunto de dados CIFAR-10, binarizará as imagens, treinará o classificador k-NN e avaliará o desempenho do modelo.

## Configurações Adicionais

- O tamanho do conjunto de dados pode ser reduzido ajustando a variável `sample_size` no script `main.py`. Isso pode ser útil para agilizar o treinamento, especialmente em computadores com recursos limitados.

- O número de vizinhos (`n_neighbors`) no classificador k-NN pode ser ajustado em `KNeighborsClassifier(n_neighbors=5)` no script `main.py`.

## Resultados

O script imprimirá a precisão do modelo no conjunto de dados de teste.

## Autor

- Ana Nicole Massaneiro
- Marcus Dudeque

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Referências

- [k-Nearest Neighbors (k-NN) Algorithm](https://didatica.tech/o-que-e-e-como-funciona-o-algoritmo-knn/)

- [Documentação oficial do Python](https://docs.python.org/3/)
