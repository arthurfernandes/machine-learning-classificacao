import pandas as pd
from collections import Counter
from collections import namedtuple

ModeloComTaxa = namedtuple("ModeloComTaxa", "nome modelo taxa")

def ler_dados():
    df = pd.read_csv('buscas.csv')

    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']

    Xdummies_df = pd.get_dummies(X_df).astype(int)
    Ydummies_df = Y_df

    X = Xdummies_df.get_values()
    Y = Ydummies_df.get_values()

    return X, Y

def separar_dados_treino_teste_real(X, Y):

    porcentagem_treino = 0.8
    porcentagem_teste = 0.1
    porcentagem_real = 0.1

    total = len(Y)
    tamanho_treino = int(porcentagem_treino * total)
    tamanho_teste = int(porcentagem_teste * total)
    tamanho_real = total - tamanho_treino - tamanho_teste

    treino_dados = X[0:tamanho_treino]
    treino_marcacoes = Y[0:tamanho_treino]
    fim_teste = tamanho_treino + tamanho_teste
    teste_dados = X[tamanho_treino:fim_teste]
    teste_marcacoes = Y[tamanho_treino:fim_teste]

    real_dados = X[fim_teste:]
    real_marcacoes = Y[fim_teste:]

    return treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, real_dados, real_marcacoes

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):

    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)

    diferencas = resultado == teste_marcacoes
    total_acertos = len([d for d in diferencas if bool(d) is True])

    total_teste = len(teste_marcacoes)
    taxa_acerto = 100.0 * total_acertos / total_teste
    print("Taxa de acerto do algoritmo %s : %f" % (nome, taxa_acerto))

    return taxa_acerto

def teste_real(nome, modelo, real_dados, real_marcacoes):
    resultado = modelo.predict(real_dados)

    diferencas = resultado == real_marcacoes
    total_acertos = len([d for d in diferencas if bool(d) is True])

    total_teste = len(teste_marcacoes)
    taxa_acerto = 100.0 * total_acertos / total_teste
    print("Dados Reais - taxa de acerto do algoritmo %s : %f" % (nome, taxa_acerto))

    return taxa_acerto

def escolher_modelo_vencedor(modelos_com_taxas):
    melhor_taxa = -1
    melhor_modelo = None
    for mct in modelos_com_taxas:
        if mct.taxa > melhor_taxa:
            melhor_taxa = mct.taxa
            melhor_modelo = mct

    return melhor_modelo


X, Y = ler_dados()

treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, real_dados, real_marcacoes = separar_dados_treino_teste_real(X, Y)

modelos_com_taxas = []

from sklearn.naive_bayes import MultinomialNB
nome = "MultinomialNB"
modelo = MultinomialNB()
taxa_acerto = fit_and_predict(nome , modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelos_com_taxas.append(ModeloComTaxa(nome, modelo, taxa_acerto))

from sklearn.ensemble import AdaBoostClassifier
nome = "AdaboostClassifier"
modelo = AdaBoostClassifier()
taxa_acerto = fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

modelos_com_taxas.append(ModeloComTaxa(nome, modelo, taxa_acerto))

mct = escolher_modelo_vencedor(modelos_com_taxas)
modelo_vencedor = mct.modelo
nome_modelo_vencedor = mct.nome

taxa_real = teste_real(nome_modelo_vencedor, modelo_vencedor, real_dados, real_marcacoes)

tamanho_teste = len(teste_marcacoes)
taxa_acerto_base = 100.0 * max(Counter(teste_marcacoes).itervalues()) / tamanho_teste

print("Taxa de acerto base: %f" % taxa_acerto_base)
print("Tamanho do teste: %d" % len(teste_marcacoes))
