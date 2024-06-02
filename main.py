import pandas as pd
import numpy as np

# Criação do modelo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Avaliação de métricas
from sklearn.metrics import accuracy_score

df = pd.read_csv('palmerpenguins.csv')

df.replace('Adelie', '0', inplace=True)
df.replace('Chinstrap', '1', inplace=True)
df.replace('Gentoo', '2', inplace=True)

df.replace('FEMALE', '0', inplace=True)
df.replace('MALE', '1', inplace=True)

df.replace('Biscoe', '0', inplace=True)
df.replace('Dream', '1', inplace=True)
df.replace('Torgersen', '2', inplace=True)

novas_colunas = ['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']
df_reindexado = df.reindex(columns=novas_colunas)


# instanciando modelos
tree = DecisionTreeClassifier()

# Variáveis preditoras
x = df_reindexado.loc[:, ['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
x = np.array(x)

# Variável alvo
y = df.loc[:, ['species']]
y = np.array(y)

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
print(f"Tamanho X de treino: {X_train.shape}")
print(f"Tamanho X de teste: {X_test.shape}")
print(f"Tamanho y de treino: {y_train.shape}")
print(f"Tamanho y de teste: {y_test.shape}")

# Treinando modelos
tree.fit(X_train, y_train)

# Prevendo valores
tree_predict = tree.predict(X_test)

# Avaliação
print()
tree_score = accuracy_score(y_test, tree_predict)
print(f"Pontuação Decision Tree:{tree_score}")
print()