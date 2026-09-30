"""Pipelines do torneio, espaços de busca, ensemble, calibração e treino final."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from autocred.avaliacao import ajustar_platt, aplicar_platt, avaliar_temporal, inclinacao_calibracao
from autocred.dados import CATEGORICAS, colunas_entrada, numericas_modelo, validar_sem_proibidas
from autocred.features import adicionar_indicadores_ausencia

SEMENTE = 42
N_CONFIGS = 25
TIPOS = ("logistica", "random_forest", "xgboost", "lightgbm")

# Cauda longa: na logística, entram em log1p.
ASSIMETRICAS = [
    "valor_bem",
    "valor_entrada",
    "valor_financiado",
    "parcela_mensal",
    "renda_mensal_declarada",
    "tempo_emprego_meses",
]

# Bom senso de crédito: +1 = risco sobe com a variável; -1 = risco cai.
MONOTONIA = {
    "qtd_restricoes_ativas": 1,
    "qtd_consultas_bureau_3m": 1,
    "ltv": 1,
    "comprometimento_renda": 1,
    "score_bureau": -1,
    "renda_mensal_declarada": -1,
    "tempo_emprego_meses": -1,
}

ESPACOS = {
    "logistica": {"C": list(np.logspace(-3, 1, N_CONFIGS))},
    "random_forest": {
        "min_samples_leaf": [20, 50, 100, 200],
        "max_features": [0.2, 0.33, 0.5, "sqrt"],
        "max_depth": [None, 6, 10, 16],
    },
    "xgboost": {
        "n_estimators": [200, 400, 700, 1000],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_child_weight": [1, 5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
    },
    "lightgbm": {
        "n_estimators": [200, 400, 700, 1000],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "num_leaves": [4, 8, 15, 31],
        "min_child_samples": [20, 50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9],
        "reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
    },
}


def amostrar_configs(tipo: str, n: int = N_CONFIGS, semente: int = SEMENTE) -> list[dict]:
    if tipo == "logistica":
        return [{"C": float(c)} for c in ESPACOS["logistica"]["C"]]
    return [dict(c) for c in ParameterSampler(ESPACOS[tipo], n_iter=n, random_state=semente)]


def restricoes_monotonicas(conjunto: str) -> list[int]:
    """Uma direção por coluna, na ordem em que o ColumnTransformer das árvores entrega."""
    return [MONOTONIA.get(c, 0) for c in numericas_modelo(conjunto)] + [0] * len(CATEGORICAS)


def _preparo(tipo: str, conjunto: str) -> ColumnTransformer:
    numericas = numericas_modelo(conjunto)
    if tipo == "logistica":
        assimetricas = [c for c in numericas if c in ASSIMETRICAS]
        demais = [c for c in numericas if c not in ASSIMETRICAS]
        return ColumnTransformer([
            ("log", Pipeline([
                ("imputa", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p)),
                ("escala", StandardScaler()),
            ]), assimetricas),
            ("num", Pipeline([
                ("imputa", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
            ]), demais),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ])
    # Árvores: ordem fixa (numéricas, depois categóricas) para as restrições monotônicas.
    num = SimpleImputer(strategy="median") if tipo == "random_forest" else "passthrough"
    return ColumnTransformer([
        ("num", num, numericas),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS),
    ])


def _classificador(tipo: str, params: dict, conjunto: str, monotonico: bool):
    if tipo == "logistica":
        return LogisticRegression(max_iter=5000, **params)
    if tipo == "random_forest":
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEMENTE, **params)
    restricoes = restricoes_monotonicas(conjunto) if monotonico else None
    if tipo == "xgboost":
        extra = {"monotone_constraints": "(" + ",".join(map(str, restricoes)) + ")"} if restricoes else {}
        return XGBClassifier(
            tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEMENTE, **extra, **params
        )
    if tipo == "lightgbm":
        extra = {"monotone_constraints": restricoes} if restricoes else {}
        return LGBMClassifier(
            subsample_freq=1, deterministic=True, force_row_wise=True, n_jobs=-1,
            random_state=SEMENTE, verbose=-1, **extra, **params,
        )
    raise ValueError(f"tipo desconhecido: {tipo}")


def construir_pipeline(tipo: str, params: dict | None = None, conjunto: str = "portatil", monotonico: bool = False) -> Pipeline:
    """Pipeline completo: indicadores de ausência → preparo → classificador. Tudo ajustado só no treino."""
    validar_sem_proibidas(colunas_entrada(conjunto))
    return Pipeline([
        ("indicadores", FunctionTransformer(adicionar_indicadores_ausencia)),
        ("preparo", _preparo(tipo, conjunto)),
        ("modelo", _classificador(tipo, params or {}, conjunto, monotonico)),
    ])


def construir_modelo(descricao: dict, conjunto: str = "portatil"):
    if descricao["tipo"] == "ensemble":
        estimadores = [(f"m{i}", construir_modelo(c, conjunto)) for i, c in enumerate(descricao["componentes"])]
        return VotingClassifier(estimadores, voting="soft")
    return construir_pipeline(descricao["tipo"], descricao.get("params"), conjunto, descricao.get("monotonico", False))


def complexidade(descricao: dict) -> int:
    if descricao["tipo"] == "ensemble":
        return 3
    if descricao["tipo"] == "logistica":
        return 0
    return 1 if descricao.get("monotonico") else 2


class ModeloCalibrado:
    """Sigmoide (Platt) sobre a PD do modelo base: muda nível e dispersão, não a ordem (nem o AUC)."""

    def __init__(self, modelo, a: float, b: float):
        self.modelo = modelo
        self.a = a
        self.b = b

    def predict_proba(self, X):
        q = aplicar_platt(self.modelo.predict_proba(X)[:, 1], self.a, self.b)
        return np.column_stack([1 - q, q])


def treinar_final(descricao: dict, X, y, anos, conjunto: str = "portatil"):
    """Mede fora do tempo, decide a calibração e retreina com a Base A inteira."""
    X = X[colunas_entrada(conjunto)]
    y = np.asarray(y)
    anos = np.asarray(anos)
    metricas, previsoes = avaliar_temporal(lambda: construir_modelo(descricao, conjunto), X, y, anos)
    inclinacao = inclinacao_calibracao(y[anos == 2024], previsoes["J2"])
    calibrar = bool(descricao["tipo"] == "random_forest" or not 0.8 <= inclinacao <= 1.2)
    modelo = construir_modelo(descricao, conjunto).fit(X, y)
    info = {**metricas, "conjunto": conjunto, "inclinacao_J2": inclinacao, "calibrado": calibrar}
    if calibrar:
        y_oot = np.concatenate([y[anos == 2023], y[anos == 2024]])
        p_oot = np.concatenate([previsoes["J1"], previsoes["J2"]])
        a, b = ajustar_platt(y_oot, p_oot)
        modelo = ModeloCalibrado(modelo, a, b)
        previsoes = {nome: aplicar_platt(p, a, b) for nome, p in previsoes.items()}
        info.update(platt_a=a, platt_b=b)
    return modelo, info, previsoes
