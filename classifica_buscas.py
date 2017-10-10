import pandas as pd
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('buscas.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.get_values()
Y = Ydummies_df.get_values()

porcentagem_treino = 0.9

total = len(Y)
tamanho_treino = int(porcentagem_treino * total)
tamanho_teste = total - tamanho_treino

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]

modelo = MultinomialNB()

modelo.fit(treino_dados, treino_marcacoes)
resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
total_acertos = len([d for d in diferencas if d == 0])

taxa_acerto = 100.0 * total_acertos / tamanho_teste

print(tamanho_teste)
print(taxa_acerto)

