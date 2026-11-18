"""Random forest para classificação, feita com as árvores de `decisiontrees.py`.

Cada árvore é treinada numa amostra das linhas tirada com reposição, do mesmo
tamanho que o treino, e em cada nó só procura a divisão entre algumas variáveis
sorteadas (`max_features`). A previsão é a média das probabilidades de todas
as árvores.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from libs.decisiontrees import DecisionTreeClassifier


def _treinar_arvore(X, y, linhas, params):
    return DecisionTreeClassifier(**params).fit(X.iloc[linhas], y[linhas])


class RandomForestClassifier:
    """Random forest de árvores CART.

    n_estimators : nº de árvores
    max_features : variáveis sorteadas em cada nó ("sqrt", int, float ou None)
    bootstrap : se cada árvore treina numa amostra com reposição (True) ou em todas as linhas
    random_state : semente das amostras e dos sorteios
    n_jobs : nº de árvores treinadas ao mesmo tempo (-1 = todos os núcleos). Não muda o resultado
    criterion, max_depth, min_samples_split, min_samples_leaf,
    min_impurity_decrease, class_weight : passados a cada árvore
    """

    PARAMETROS = ("n_estimators", "max_features", "bootstrap", "random_state", "n_jobs", "criterion",
                  "max_depth", "min_samples_split", "min_samples_leaf", "min_impurity_decrease", "class_weight")

    def __init__(self, n_estimators=100, max_features="sqrt", bootstrap=True, random_state=None, n_jobs=1,
                 criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, class_weight=None):
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    def get_params(self):
        return {p: getattr(self, p) for p in self.PARAMETROS}

    def fit(self, X, y):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.feature_names_ = list(X.columns)
        rng = np.random.default_rng(self.random_state)

        # Os pesos "balanced" vêm do treino inteiro, para todas as árvores usarem os mesmos
        pesos = self.class_weight
        if pesos == "balanced":
            contagem = pd.Series(y).value_counts()
            pesos = (len(y) / (len(contagem) * contagem)).to_dict()

        # As amostras e as sementes são sorteadas antes, para o resultado não depender de n_jobs
        tarefas = []
        for _ in range(self.n_estimators):
            linhas = rng.integers(0, len(y), len(y)) if self.bootstrap else np.arange(len(y))
            params = dict(criterion=self.criterion, max_depth=self.max_depth,
                          min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                          min_impurity_decrease=self.min_impurity_decrease, class_weight=pesos,
                          max_features=self.max_features, random_state=int(rng.integers(2**31)))
            tarefas.append((linhas, params))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_treinar_arvore)(X, y, linhas, params) for linhas, params in tarefas)
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        for arvore in self.estimators_:
            # Uma amostra pode não ter todas as classes, por isso alinhamos as colunas
            colunas = np.searchsorted(self.classes_, arvore.classes_)
            proba[:, colunas] += arvore.predict_proba(X)
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def feature_importances(self):
        """Média das importâncias das árvores, da mais para a menos importante."""
        media = np.mean([a.feature_importances_ for a in self.estimators_], axis=0)
        return pd.Series(media, index=self.feature_names_).sort_values(ascending=False)
