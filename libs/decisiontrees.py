"""Árvore de decisão para classificação, implementada de raiz.

Só usa numpy/pandas para guardar e somar arrays; o algoritmo (escolha das
divisões, crescimento da árvore, previsão) está todo aqui.

Características:
- Critério de impureza: Gini ou entropia.
- Divisões binárias:
    * numéricas por limiar   -> `x <= t` vs `x > t`
    * categóricas uma-contra-todas -> `x = c` vs `x ≠ c`
- Valores em falta (NaN): em cada divisão testa-se mandar as linhas em falta
  para a esquerda e para a direita, e fica a opção com menor impureza.
  Se no treino não havia faltas nesse nó, vão para o filho com mais peso.
- Hiperparâmetros: max_depth, min_samples_split, min_samples_leaf,
  min_impurity_decrease.
- Pesos por classe (`class_weight="balanced"` ou dicionário) e por amostra.
- Importância das variáveis (redução de impureza ponderada) e exportação das
  regras em texto.

API ao estilo do scikit-learn: fit, predict, predict_proba, score,
get_params, set_params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_EPS = 1e-12


def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Impureza de uma (ou várias) distribuições de pesos por classe.

    `counts` tem forma (..., K); devolve um array com forma (...).
    """
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts, dtype=float), where=total > 0)
    if criterion == "gini":
        return 1.0 - (p ** 2).sum(axis=-1)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logp).sum(axis=-1)


class Node:
    """Nó da árvore. É uma folha quando `feature is None`."""

    __slots__ = (
        "value", "class_counts", "n_samples", "impurity", "depth",
        "feature", "threshold", "category", "missing_left", "missing_seen",
        "left", "right",
    )

    def __init__(self, value, class_counts, n_samples, impurity, depth):
        self.value = value                # pesos por classe (com class_weight)
        self.class_counts = class_counts  # contagens reais por classe
        self.n_samples = n_samples
        self.impurity = impurity
        self.depth = depth
        self.feature = None      # índice da variável usada na divisão
        self.threshold = None    # limiar (variável numérica)
        self.category = None     # código da categoria (variável categórica)
        self.missing_left = True  # para onde vão os valores em falta
        self.missing_seen = False  # se havia faltas neste nó durante o treino
        self.left = None
        self.right = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """Árvore de decisão para classificação (CART com divisões binárias).

    Parâmetros
    ----------
    criterion : "gini" ou "entropy"
    max_depth : profundidade máxima (None = sem limite)
    min_samples_split : nº mínimo de amostras num nó para o tentar dividir
    min_samples_leaf : nº mínimo de amostras em cada folha
    min_impurity_decrease : redução mínima de impureza (ponderada pela fração
        do peso total que chega ao nó) para aceitar uma divisão
    class_weight : None, "balanced" ou dicionário {classe: peso}
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        class_weight: str | dict | None = None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight

    # ------------------------------------------------------------------
    # Parâmetros
    # ------------------------------------------------------------------
    _PARAMS = (
        "criterion", "max_depth", "min_samples_split", "min_samples_leaf",
        "min_impurity_decrease", "class_weight",
    )

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in self._PARAMS}

    def set_params(self, **params) -> "DecisionTreeClassifier":
        for k, v in params.items():
            if k not in self._PARAMS:
                raise ValueError(f"Parâmetro desconhecido: {k}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"DecisionTreeClassifier({args})"

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None) -> "DecisionTreeClassifier":
        if self.criterion not in ("gini", "entropy"):
            raise ValueError("criterion tem de ser 'gini' ou 'entropy'")
        if self.min_samples_leaf < 1 or self.min_samples_split < 2:
            raise ValueError("min_samples_leaf >= 1 e min_samples_split >= 2")

        X = self._as_frame(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X e y têm tamanhos diferentes")
        if pd.isna(y).any():
            raise ValueError("y tem valores em falta")

        n = len(y)
        self.feature_names_ = list(X.columns)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)

        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w * self._class_weights(y_idx, K)[y_idx]

        # Matriz (n, K) com o peso de cada amostra na coluna da sua classe
        self._Yw = np.zeros((n, K))
        self._Yw[np.arange(n), y_idx] = w
        self._y_idx = y_idx

        # Cada coluna fica guardada como float (numérica) ou códigos (categórica)
        self.feature_types_ = []
        self.categories_ = {}
        self._cols = []
        for name in self.feature_names_:
            s = X[name]
            if is_numeric_dtype(s) and not is_bool_dtype(s):
                self.feature_types_.append("num")
                self._cols.append(s.to_numpy(dtype=float))
            else:
                cat = pd.Categorical(s)
                self.feature_types_.append("cat")
                self.categories_[name] = list(cat.categories)
                self._cols.append(np.asarray(cat.codes, dtype=np.int64))  # NaN -> -1

        self._total_weight = w.sum()
        self._importances = np.zeros(len(self.feature_names_))
        self.tree_ = self._grow(np.arange(n), depth=0)

        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        del self._Yw, self._y_idx, self._cols, self._importances
        return self

    def _class_weights(self, y_idx: np.ndarray, K: int) -> np.ndarray:
        if self.class_weight is None:
            return np.ones(K)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=K)
            return len(y_idx) / (K * counts)
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(c, 1.0) for c in self.classes_], dtype=float)
        raise ValueError("class_weight tem de ser None, 'balanced' ou um dicionário")

    def _grow(self, idx: np.ndarray, depth: int) -> Node:
        counts = self._Yw[idx].sum(axis=0)
        class_counts = np.bincount(self._y_idx[idx], minlength=len(self.classes_))
        imp = float(_impurity(counts, self.criterion))
        node = Node(counts, class_counts, len(idx), imp, depth)

        # Critérios de paragem
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(idx) < self.min_samples_split
            or len(idx) < 2 * self.min_samples_leaf
            or imp <= _EPS
        ):
            return node

        split = self._best_split(idx)
        if split is None:
            return node
        child_imp, f, threshold, category, missing_left, missing_seen, go_left = split

        gain = imp - child_imp
        W = counts.sum()
        if gain <= _EPS or (W / self._total_weight) * gain < self.min_impurity_decrease:
            return node

        self._importances[f] += W * gain
        node.feature = f
        node.threshold = threshold
        node.category = category
        node.missing_left = missing_left
        node.missing_seen = missing_seen
        node.left = self._grow(idx[go_left], depth + 1)
        node.right = self._grow(idx[~go_left], depth + 1)
        return node

    def _best_split(self, idx: np.ndarray):
        """Procura a melhor divisão do nó entre todas as variáveis.

        Devolve (impureza dos filhos, variável, limiar, categoria,
        faltas à esquerda?, havia faltas?, máscara das linhas que vão para a
        esquerda) ou None se não houver nenhuma divisão válida.
        """
        Yw = self._Yw[idx]
        best_imp = np.inf
        best = None

        for f, kind in enumerate(self.feature_types_):
            x = self._cols[f][idx]
            miss = np.isnan(x) if kind == "num" else (x == -1)
            n_miss = int(miss.sum())
            pres = ~miss
            miss_counts = Yw[miss].sum(axis=0)

            if kind == "num":
                xp = x[pres]
                if xp.size < 2:
                    continue
                order = np.argsort(xp, kind="mergesort")
                xs = xp[order]
                cum = np.cumsum(Yw[pres][order], axis=0)
                # Só faz sentido cortar entre dois valores diferentes
                pos = np.nonzero(xs[:-1] < xs[1:])[0]
                if pos.size == 0:
                    continue
                left = cum[pos]
                right = cum[-1] - left
                n_left = pos + 1
                n_right = xs.size - n_left
                candidates = (xs[pos] + xs[pos + 1]) / 2  # limiares
            else:
                codes = x[pres]
                ncat = len(self.categories_[self.feature_names_[f]])
                n_by_cat = np.bincount(codes, minlength=ncat)
                cats = np.nonzero(n_by_cat > 0)[0]
                if cats.size < 2:
                    continue
                Yp = Yw[pres]
                by_cat = np.column_stack(
                    [np.bincount(codes, weights=Yp[:, k], minlength=ncat) for k in range(Yw.shape[1])]
                )
                left = by_cat[cats]
                right = by_cat.sum(axis=0) - left
                n_left = n_by_cat[cats]
                n_right = codes.size - n_left
                candidates = cats  # categorias

            # Testar faltas à esquerda e à direita (se existirem)
            for miss_left in ((True, False) if n_miss else (True,)):
                L = left + miss_counts if miss_left else left
                R = right if miss_left else right + miss_counts
                nL = n_left + (n_miss if miss_left else 0)
                nR = n_right + (0 if miss_left else n_miss)
                ok = (nL >= self.min_samples_leaf) & (nR >= self.min_samples_leaf)
                if not ok.any():
                    continue
                WL, WR = L.sum(axis=1), R.sum(axis=1)
                child = (WL * _impurity(L, self.criterion) + WR * _impurity(R, self.criterion)) / (WL + WR)
                child[~ok] = np.inf
                j = int(np.argmin(child))
                if child[j] < best_imp - _EPS:
                    best_imp = child[j]
                    # Sem faltas no treino: no futuro vão para o filho com mais peso
                    ml = miss_left if n_miss else bool(WL[j] >= WR[j])
                    best = (f, kind, candidates[j], ml, n_miss > 0)

        if best is None:
            return None
        f, kind, cand, miss_left, missing_seen = best
        x = self._cols[f][idx]
        if kind == "num":
            miss = np.isnan(x)
            go_left = np.zeros(len(idx), dtype=bool)
            go_left[~miss] = x[~miss] <= cand
            threshold, category = float(cand), None
        else:
            miss = x == -1
            go_left = x == cand
            threshold, category = None, int(cand)
        go_left[miss] = miss_left
        return best_imp, f, threshold, category, miss_left, missing_seen, go_left

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    def _prepare(self, X) -> list[np.ndarray]:
        X = self._as_frame(X)
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Faltam colunas em X: {missing}")
        cols = []
        for name, kind in zip(self.feature_names_, self.feature_types_):
            s = X[name]
            if kind == "num":
                cols.append(s.to_numpy(dtype=float))
            else:
                codes = np.asarray(pd.Categorical(s, categories=self.categories_[name]).codes, dtype=np.int64)
                # Categoria nunca vista no treino: -2 (não é "em falta", vai sempre para o lado ≠)
                codes[(codes == -1) & s.notna().to_numpy()] = -2
                cols.append(codes)
        return cols

    def predict_proba(self, X) -> np.ndarray:
        cols = self._prepare(X)
        n = len(cols[0]) if cols else len(X)
        out = np.zeros((n, len(self.classes_)))
        self._route(self.tree_, np.arange(n), cols, out)
        return out

    def _route(self, node: Node, idx: np.ndarray, cols, out: np.ndarray) -> None:
        if idx.size == 0:
            return
        if node.is_leaf:
            total = node.value.sum()
            out[idx] = node.value / total if total > 0 else 1.0 / len(self.classes_)
            return
        x = cols[node.feature][idx]
        if self.feature_types_[node.feature] == "num":
            miss = np.isnan(x)
            go_left = np.zeros(idx.size, dtype=bool)
            go_left[~miss] = x[~miss] <= node.threshold
        else:
            miss = x == -1
            go_left = x == node.category
        go_left[miss] = node.missing_left
        self._route(node.left, idx[go_left], cols, out)
        self._route(node.right, idx[~go_left], cols, out)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------
    # Inspeção
    # ------------------------------------------------------------------
    def get_depth(self) -> int:
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.tree_)

    def get_n_leaves(self) -> int:
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return leaves(self.tree_)

    def feature_importances(self) -> pd.Series:
        """Importâncias como Series ordenada (da mais para a menos importante)."""
        return pd.Series(self.feature_importances_, index=self.feature_names_).sort_values(ascending=False)

    def _conditions(self, node: Node, decimals: int) -> tuple[str, str]:
        """Texto da condição para o ramo esquerdo e para o direito."""
        name = self.feature_names_[node.feature]
        if node.threshold is not None:
            left = f"{name} <= {node.threshold:.{decimals}f}"
            right = f"{name} > {node.threshold:.{decimals}f}"
        else:
            cat = self.categories_[name][node.category]
            left, right = f"{name} = {cat}", f"{name} ≠ {cat}"
        if node.missing_seen:
            if node.missing_left:
                left += " ou em falta"
            else:
                right += " ou em falta"
        return left, right

    def _leaf_text(self, node: Node) -> str:
        pred = self.classes_[int(np.argmax(node.value))]
        dist = ", ".join(
            f"{c}: {k / node.n_samples:.0%}" for c, k in zip(self.classes_, node.class_counts)
        )
        return f"→ {pred}  (n={node.n_samples}; {dist})"

    def export_text(self, max_depth: int | None = None, decimals: int = 1) -> str:
        """Árvore em texto, com uma linha por condição e por folha."""
        lines = []

        def walk(node, indent):
            if node.is_leaf or (max_depth is not None and node.depth >= max_depth):
                suffix = "" if node.is_leaf else "  [...]"
                lines.append(f"{indent}{self._leaf_text(node)}{suffix}")
                return
            left, right = self._conditions(node, decimals)
            lines.append(f"{indent}{left}")
            walk(node.left, indent + "|   ")
            lines.append(f"{indent}{right}")
            walk(node.right, indent + "|   ")

        walk(self.tree_, "")
        return "\n".join(lines)

    def rules(self, decimals: int = 1) -> pd.DataFrame:
        """Uma linha por folha: regra completa, classe prevista, suporte e distribuição."""
        rows = []

        def walk(node, conds):
            if node.is_leaf:
                k = int(np.argmax(node.value))
                row = {
                    "regra": " E ".join(conds) if conds else "(todas)",
                    "classe": self.classes_[k],
                    "n": node.n_samples,
                    "pureza": node.class_counts[k] / node.n_samples,
                }
                for c, cnt in zip(self.classes_, node.class_counts):
                    row[f"%{c}"] = cnt / node.n_samples
                rows.append(row)
                return
            left, right = self._conditions(node, decimals)
            walk(node.left, conds + [left])
            walk(node.right, conds + [right])

        walk(self.tree_, [])
        return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X tem de ser 2D")
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
