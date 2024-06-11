import pandas as pd
import numpy as np

# Criação do modelo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
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

predict = tree.predict(X_test)

relatorio_classificacao = classification_report(y_test, predict, target_names=['Adelie', 'Chinstrap', 'Gentoo'])
print("Relatório de Classificação:\n", relatorio_classificacao)

def classificar_novos_dados(tree):
    print("Insira os seguintes dados para classificação:")
    island = int(input("Ilha (0 para Biscoe, 1 para Dream, 2 para Torgersen): "))
    sex = int(input("Sexo (0 para FEMALE, 1 para MALE): "))
    culmen_length_mm = float(input("Comprimento do culmen (mm): "))
    culmen_depth_mm = float(input("Profundidade do culmen (mm): "))
    flipper_length_mm = float(input("Comprimento da nadadeira (mm): "))
    body_mass_g = float(input("Massa corporal (g): "))

    # Criar um DataFrame com os dados inseridos
    novos_dados = pd.DataFrame([[island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]],
                               columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',
                                        'body_mass_g'])

    predicao = tree.predict(novos_dados)
    especies = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

    print(f"A previsão do modelo é que o pinguim pertence à espécie: {especies[predicao[0]]}")


classificar_novos_dados(tree)
