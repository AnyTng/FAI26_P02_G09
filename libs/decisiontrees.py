"""Árvore de decisão para classificação, implementada de raiz.

numpy e pandas servem só para guardar e somar arrays: a escolha das divisões,
o crescimento da árvore e a previsão estão todos aqui.

Todas as variáveis têm de ser numéricas e sem valores em falta. O notebook
trata disso antes de treinar a árvore.

Hiperparâmetros: criterion (gini/entropy), max_depth, min_samples_split,
min_samples_leaf, min_impurity_decrease, class_weight e max_features.
"""

import numpy as np
import pandas as pd


def impureza(pesos, criterio):
    """Impureza de uma ou várias distribuições de classes.

    `pesos` tem forma (..., K): o peso de cada classe. Devolve forma (...).
    """
    total = pesos.sum(axis=-1, keepdims=True)
    p = np.divide(pesos, total, out=np.zeros_like(pesos, dtype=float), where=total > 0)
    if criterio == "gini":
        return 1 - (p ** 2).sum(axis=-1)
    log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * log_p).sum(axis=-1)


class No:
    """Nó da árvore. É uma folha quando `variavel is None`."""

    def __init__(self, pesos, contagens, profundidade):
        self.pesos = pesos            # peso de cada classe (com class_weight)
        self.contagens = contagens    # nº real de amostras de cada classe
        self.n = int(contagens.sum())
        self.profundidade = profundidade
        self.variavel = None          # índice da variável da divisão
        self.limiar = None            # vai para a esquerda se x <= limiar
        self.esquerda = None
        self.direita = None

    @property
    def folha(self):
        return self.variavel is None


class DecisionTreeClassifier:
    """Árvore de decisão (CART, divisões binárias `x <= limiar`).

    criterion : "gini" ou "entropy"
    max_depth : profundidade máxima (None = sem limite)
    min_samples_split : nº mínimo de amostras para tentar dividir um nó
    min_samples_leaf : nº mínimo de amostras em cada folha
    min_impurity_decrease : ganho mínimo (ponderado pela fração do peso total
        que chega ao nó) para aceitar uma divisão
    class_weight : None, "balanced" ou {classe: peso}
    max_features : nº de variáveis sorteadas em cada nó para procurar a divisão.
        None = todas, "sqrt" = raiz do nº de variáveis, int = esse número,
        float = essa fração. É o que a random forest usa para diversificar as árvores.
    random_state : semente do sorteio das variáveis (só conta com max_features)
    """

    PARAMETROS = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf",
                  "min_impurity_decrease", "class_weight", "max_features", "random_state")

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, class_weight=None,
                 max_features=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.max_features = max_features
        self.random_state = random_state

    def get_params(self):
        return {p: getattr(self, p) for p in self.PARAMETROS}

    def set_params(self, **params):
        for p, v in params.items():
            if p not in self.PARAMETROS:
                raise ValueError(f"Parâmetro desconhecido: {p}")
            setattr(self, p, v)
        return self

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def fit(self, X, y):
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(f'criterion tem de ser "gini" ou "entropy", não {self.criterion!r}')
        X, nomes = self._matriz(X)
        y = np.asarray(y)
        self.feature_names_ = nomes
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)

        # Peso de cada amostra, conforme a sua classe
        peso_classe = self._pesos_classes(y_idx, K)
        # Y[i, k] = peso da amostra i se for da classe k, senão 0
        self._Y = np.zeros((len(y), K))
        self._Y[np.arange(len(y)), y_idx] = peso_classe[y_idx]
        self._y_idx = y_idx
        self._X = X
        self._peso_total = self._Y.sum()
        self._ganhos = np.zeros(X.shape[1])
        self._n_sorteadas = self._numero_variaveis(X.shape[1])
        self._rng = np.random.default_rng(self.random_state)

        self.tree_ = self._crescer(np.arange(len(y)), profundidade=0)

        # Importância = ganho de impureza acumulado por variável, normalizado
        total = self._ganhos.sum()
        self.feature_importances_ = self._ganhos / total if total > 0 else self._ganhos
        # Variáveis 0/1 aparecem nas regras como "= 1" / "= 0"
        self._binarias = [set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(X.shape[1])]
        del self._X, self._Y, self._y_idx, self._ganhos, self._rng
        return self

    def _numero_variaveis(self, p):
        """Quantas variáveis se testam em cada nó, conforme max_features."""
        if self.max_features is None:
            return p
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(p)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * p))
        return min(p, int(self.max_features))

    def _pesos_classes(self, y_idx, K):
        if self.class_weight is None:
            return np.ones(K)
        if self.class_weight == "balanced":
            # Cada classe passa a pesar o mesmo no total: n / (K * n_classe)
            return len(y_idx) / (K * np.bincount(y_idx, minlength=K))
        return np.array([self.class_weight.get(c, 1.0) for c in self.classes_], dtype=float)

    def _crescer(self, idx, profundidade):
        pesos = self._Y[idx].sum(axis=0)
        contagens = np.bincount(self._y_idx[idx], minlength=len(self.classes_))
        no = No(pesos, contagens, profundidade)
        imp = impureza(pesos, self.criterion)

        # Critérios de paragem
        if ((self.max_depth is not None and profundidade >= self.max_depth)
                or len(idx) < self.min_samples_split
                or len(idx) < 2 * self.min_samples_leaf
                or imp == 0):
            return no

        divisao = self._melhor_divisao(idx)
        if divisao is None:
            return no
        imp_filhos, j, limiar = divisao

        # Ganho de impureza, ponderado pela fração do peso que chega a este nó.
        # Aceitamos divisões com ganho 0 porque podem permitir boas divisões mais abaixo,
        # como no XOR. O 1e-12 evita que um ganho 0 com erro de arredondamento conte como negativo
        ganho = max(imp - imp_filhos, 0.0)
        fracao = pesos.sum() / self._peso_total
        if fracao * ganho + 1e-12 < self.min_impurity_decrease:
            return no

        self._ganhos[j] += pesos.sum() * ganho
        no.variavel, no.limiar = j, limiar
        vai_esq = self._X[idx, j] <= limiar
        no.esquerda = self._crescer(idx[vai_esq], profundidade + 1)
        no.direita = self._crescer(idx[~vai_esq], profundidade + 1)
        return no

    def _melhor_divisao(self, idx):
        """Testa todos os limiares de todas as variáveis e devolve o melhor.

        Para uma variável, ordenam-se as amostras pelo seu valor. Cortar entre
        a posição i e i+1 deixa as primeiras i+1 amostras à esquerda, por isso
        os pesos por classe da esquerda são uma soma acumulada (cumsum) e os
        da direita são o total menos a esquerda. Assim avaliam-se todos os
        cortes de uma vez, sem um ciclo por limiar.

        Devolve (impureza ponderada dos filhos, variável, limiar) ou None.
        """
        Y = self._Y[idx]
        n = len(idx)
        msl = self.min_samples_leaf
        melhor = None

        p = self._X.shape[1]
        if self._n_sorteadas < p:
            variaveis = self._rng.choice(p, self._n_sorteadas, replace=False)
        else:
            variaveis = range(p)

        for j in variaveis:
            x = self._X[idx, j]
            ordem = np.argsort(x, kind="mergesort")
            x_ord = x[ordem]

            acumulado = np.cumsum(Y[ordem], axis=0)
            esq = acumulado[:-1]                     # pesos à esquerda em cada corte
            dir_ = acumulado[-1] - esq               # total - esquerda
            n_esq = np.arange(1, n)                  # nº de amostras à esquerda

            # Cortes válidos: entre dois valores diferentes e com folhas grandes o suficiente
            valido = (x_ord[:-1] < x_ord[1:]) & (n_esq >= msl) & (n - n_esq >= msl)
            if not valido.any():
                continue

            p_esq, p_dir = esq.sum(axis=1), dir_.sum(axis=1)
            imp_filhos = (p_esq * impureza(esq, self.criterion)
                          + p_dir * impureza(dir_, self.criterion)) / (p_esq + p_dir)
            imp_filhos[~valido] = np.inf

            i = int(np.argmin(imp_filhos))
            if melhor is None or imp_filhos[i] < melhor[0]:
                limiar = (x_ord[i] + x_ord[i + 1]) / 2   # ponto médio entre os dois valores
                melhor = (imp_filhos[i], j, limiar)
        return melhor

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X, _ = self._matriz(X, nomes_esperados=self.feature_names_)
        proba = np.zeros((len(X), len(self.classes_)))

        # Desce todas as linhas ao mesmo tempo: em cada nó separa os índices
        # que vão para a esquerda e para a direita
        def descer(no, idx):
            if no.folha:
                proba[idx] = no.pesos / no.pesos.sum()
                return
            vai_esq = X[idx, no.variavel] <= no.limiar
            descer(no.esquerda, idx[vai_esq])
            descer(no.direita, idx[~vai_esq])

        descer(self.tree_, np.arange(len(X)))
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        """Accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------
    # Inspeção
    # ------------------------------------------------------------------
    def get_depth(self):
        def prof(no):
            return 0 if no.folha else 1 + max(prof(no.esquerda), prof(no.direita))
        return prof(self.tree_)

    def get_n_leaves(self):
        def folhas(no):
            return 1 if no.folha else folhas(no.esquerda) + folhas(no.direita)
        return folhas(self.tree_)

    def feature_importances(self):
        """Importâncias como Series, da mais para a menos importante."""
        return pd.Series(self.feature_importances_, index=self.feature_names_).sort_values(ascending=False)

    def _condicoes(self, no, casas):
        nome = self.feature_names_[no.variavel]
        if self._binarias[no.variavel]:
            return f"{nome} = 0", f"{nome} = 1"
        return f"{nome} <= {no.limiar:.{casas}f}", f"{nome} > {no.limiar:.{casas}f}"

    def _texto_folha(self, no):
        classe = self.classes_[np.argmax(no.pesos)]
        dist = ", ".join(f"{c}: {k / no.n:.0%}" for c, k in zip(self.classes_, no.contagens))
        return f"→ {classe}  (n={no.n}; {dist})"

    def export_text(self, max_depth=None, decimals=1):
        """Árvore em texto. Com `max_depth`, os nós mais fundos aparecem resumidos."""
        linhas = []

        def percorrer(no, recuo):
            if no.folha or (max_depth is not None and no.profundidade >= max_depth):
                linhas.append(recuo + self._texto_folha(no) + ("" if no.folha else "  [...]"))
                return
            esq, dir_ = self._condicoes(no, decimals)
            linhas.append(recuo + esq)
            percorrer(no.esquerda, recuo + "|   ")
            linhas.append(recuo + dir_)
            percorrer(no.direita, recuo + "|   ")

        percorrer(self.tree_, "")
        return "\n".join(linhas)

    def rules(self, decimals=1):
        """Uma linha por folha: regra, classe prevista, nº de amostras, pureza
        (fração de amostras da classe prevista) e distribuição das classes.

        As condições sobre a mesma variável são juntadas num intervalo
        (ex.: `Landsize <= 1104.5 E Landsize <= 685.5` -> `Landsize <= 685.5`).
        """
        linhas = []

        def percorrer(no, limites):
            # limites = {variável: (mínimo exclusivo, máximo inclusivo)}
            if no.folha:
                k = int(np.argmax(no.pesos))
                linha = {"regra": self._texto_regra(limites, decimals),
                         "classe": self.classes_[k], "n": no.n,
                         "pureza": round(no.contagens[k] / no.n, 3)}  # % da classe prevista
                for c, cont in zip(self.classes_, no.contagens):
                    linha[f"%{c}"] = round(cont / no.n, 3)
                linhas.append(linha)
                return
            j = no.variavel
            minimo, maximo = limites.get(j, (-np.inf, np.inf))
            percorrer(no.esquerda, {**limites, j: (minimo, min(maximo, no.limiar))})
            percorrer(no.direita, {**limites, j: (max(minimo, no.limiar), maximo)})

        percorrer(self.tree_, {})
        return pd.DataFrame(linhas).sort_values("n", ascending=False, ignore_index=True)

    def _texto_regra(self, limites, casas):
        partes = []
        for j, (minimo, maximo) in limites.items():
            nome = self.feature_names_[j]
            if self._binarias[j]:
                partes.append(f"{nome} = {0 if maximo < np.inf else 1}")
            elif minimo == -np.inf:
                partes.append(f"{nome} <= {maximo:.{casas}f}")
            elif maximo == np.inf:
                partes.append(f"{nome} > {minimo:.{casas}f}")
            else:
                partes.append(f"{minimo:.{casas}f} < {nome} <= {maximo:.{casas}f}")
        return " E ".join(partes) or "(todas)"

    # ------------------------------------------------------------------
    @staticmethod
    def _matriz(X, nomes_esperados=None):
        """Converte X numa matriz float e valida-a (numérica, sem NaN)."""
        if isinstance(X, pd.DataFrame):
            if nomes_esperados is not None:
                X = X[nomes_esperados]
            nomes = list(X.columns)
            X = X.to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            nomes = nomes_esperados or [f"x{j}" for j in range(X.shape[1])]
        if np.isnan(X).any():
            raise ValueError("X tem valores em falta, preencher antes de usar a árvore")
        return X, nomes
