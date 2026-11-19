"""K-Means e métricas de agrupamento, implementados de raiz.

numpy e pandas servem só para guardar e somar arrays: a escolha dos centros,
as iterações, a silhouette e o índice de Rand ajustado estão todos aqui.

As variáveis têm de ser numéricas, sem valores em falta e já na mesma escala.
O notebook trata disso antes de correr o K-Means.
"""

import numpy as np
import pandas as pd


def distancias_quadradas(X, centros):
    """Distância ao quadrado de cada ponto a cada centro, uma coluna por centro."""
    return np.column_stack([np.sum((X - centro) ** 2, axis=1) for centro in centros])


class KMeans:
    """K-Means com distância euclidiana.

    n_clusters : nº de clusters K
    n_init : nº de tentativas com centros iniciais diferentes. Fica a de menor inércia
    max_iter : nº máximo de iterações em cada tentativa
    tol : deslocamento máximo dos centros para dar a tentativa por terminada
    random_state : semente do sorteio dos centros iniciais
    """

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        # Os centros iniciais são sorteados entre pontos diferentes, para não começar com dois centros iguais
        distintos = np.unique(X, axis=0)

        melhor = None
        tentativas = []
        for tentativa in range(self.n_init):
            iniciais = distintos[rng.choice(len(distintos), self.n_clusters, replace=False)]
            centros, grupos, inercia, iteracoes, convergiu = self._uma_tentativa(X, iniciais.copy())
            tentativas.append({"tentativa": tentativa, "inercia": inercia,
                               "iteracoes": iteracoes, "convergiu": convergiu})
            if convergiu and (melhor is None or inercia < melhor[2]):
                melhor = (centros, grupos, inercia, iteracoes)

        if melhor is None:
            raise RuntimeError("Nenhuma tentativa convergiu. Aumenta max_iter.")
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = melhor
        self.tentativas_ = pd.DataFrame(tentativas)
        return self

    def _uma_tentativa(self, X, centros):
        k = self.n_clusters
        for iteracao in range(1, self.max_iter + 1):
            distancias = distancias_quadradas(X, centros)
            grupos = distancias.argmin(axis=1)

            novos = centros.copy()
            vazios = []
            for grupo in range(k):
                membros = X[grupos == grupo]
                if len(membros):
                    novos[grupo] = membros.mean(axis=0)
                else:
                    vazios.append(grupo)

            # Um cluster vazio passa para o ponto mais afastado dos centros ocupados
            if vazios:
                ocupados = [g for g in range(k) if g not in vazios]
                proximidade = distancias_quadradas(X, novos[ocupados]).min(axis=1)
                for grupo in vazios:
                    indice = int(proximidade.argmax())
                    novos[grupo] = X[indice]
                    # Atualizamos as distâncias para não dar o mesmo ponto a dois clusters vazios
                    proximidade = np.minimum(proximidade, distancias_quadradas(X, [X[indice]])[:, 0])

            deslocamento = np.linalg.norm(novos - centros, axis=1).max()
            centros = novos
            distancias = distancias_quadradas(X, centros)
            novos_grupos = distancias.argmin(axis=1)
            inercia = float(distancias[np.arange(len(X)), novos_grupos].sum())

            # Paramos quando os centros quase não mexem e os grupos já não mudam
            if not vazios and deslocamento <= self.tol and np.array_equal(grupos, novos_grupos):
                return centros, novos_grupos, inercia, iteracao, True
        return centros, novos_grupos, inercia, iteracao, False

    def predict(self, X):
        return distancias_quadradas(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)


def silhouette(X, grupos):
    """Silhouette média de todos os pontos."""
    X = np.asarray(X, dtype=float)
    _, grupos = np.unique(grupos, return_inverse=True)
    # A silhouette só está definida com 2 clusters ou mais e menos clusters do que pontos
    k = grupos.max() + 1
    if not 2 <= k <= len(X) - 1:
        raise ValueError(f"A silhouette precisa de 2 a {len(X) - 1} clusters, recebeu {k}")
    distancias = np.sqrt(distancias_quadradas(X, X))

    # Soma das distâncias de cada ponto a todos os pontos de cada cluster
    pertenca = np.eye(grupos.max() + 1)[grupos]
    somas = distancias @ pertenca
    tamanhos = pertenca.sum(axis=0)

    linhas = np.arange(len(X))
    proprio = tamanhos[grupos]
    a = np.divide(somas[linhas, grupos], proprio - 1, out=np.zeros(len(X)), where=proprio > 1)
    medias = somas / tamanhos
    medias[linhas, grupos] = np.inf
    b = medias.min(axis=1)

    maior = np.maximum(a, b)
    valores = np.divide(b - a, maior, out=np.zeros(len(X)), where=maior > 0)
    # Um ponto sozinho no seu cluster conta como 0
    valores[proprio == 1] = 0.0
    return float(valores.mean())


def rand_ajustado(grupos_a, grupos_b):
    """Índice de Rand ajustado entre duas partições dos mesmos pontos."""
    _, a = np.unique(grupos_a, return_inverse=True)
    _, b = np.unique(grupos_b, return_inverse=True)
    tabela = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(tabela, (a, b), 1)

    pares = lambda n: n * (n - 1) / 2
    juntos = pares(tabela).sum()
    linhas = pares(tabela.sum(axis=1)).sum()
    colunas = pares(tabela.sum(axis=0)).sum()
    acaso = linhas * colunas / pares(len(a))
    denominador = (linhas + colunas) / 2 - acaso
    return 1.0 if abs(denominador) < 1e-12 else float((juntos - acaso) / denominador)
