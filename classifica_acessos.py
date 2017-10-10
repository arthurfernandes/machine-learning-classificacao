from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

def particionar_treino_testes(X, Y, porcentagem_testes):
    total = len(X)
    if total < 5:
        raise "E preciso mais de 5 amostras para particionar"

    num_testes = int(len(X) * porcentagem_testes)
    num_treino = total - num_testes

    treinoX = X[:num_treino]
    treinoY = Y[:num_treino]

    testeX = X[-num_testes:]
    testeY = Y[-num_testes:]

    return treinoX, treinoY, testeX, testeY

X, Y = carregar_acessos()

treino_dados, treino_marcacoes, teste_dados, teste_marcacoes = particionar_treino_testes(X, Y, 0.1)

modelo = MultinomialNB()

modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes
total_acertos = len([d for d in diferencas if d == 0])
total_testes = len(teste_marcacoes)
taxa_acerto = 100.0 * total_acertos / total_testes

print(resultado)
print(diferencas)
print(total_testes)
print(taxa_acerto)

