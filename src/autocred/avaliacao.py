"""Métricas de discriminação, validação temporal, escolha do campeão e calibração."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

JANELAS = {
    "J1": ((2022,), (2023,)),
    "J2": ((2022, 2023), (2024,)),
}
LIMIAR_COMPLEXIDADE = 0.005


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def ks(y, p) -> float:
    """Maior distância entre as distribuições acumuladas de maus e bons."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(referencia, atual, n_faixas: int = 10, piso: float = 1e-6) -> float:
    """Population Stability Index entre duas distribuições de score (inteiros de 1 a n_faixas).

    Leitura usual: abaixo de 0,10 estável; de 0,10 a 0,25 atenção; acima de 0,25 drift severo.
    """
    faixas = np.arange(1, n_faixas + 1)
    r = np.clip([np.mean(np.asarray(referencia) == f) for f in faixas], piso, None)
    a = np.clip([np.mean(np.asarray(atual) == f) for f in faixas], piso, None)
    return float(np.sum((a - r) * np.log(a / r)))


def bootstrap_auc(y, p, n: int = 1000, semente: int = 42) -> tuple[float, float]:
    """Intervalo de 95% do AUC por reamostragem com reposição."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(semente)
    valores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            valores.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def bootstrap_diferenca_auc(y, p1, p2, n: int = 1000, semente: int = 42) -> tuple[float, float, float]:
    """AUC(p1) − AUC(p2) nas mesmas reamostragens: média e intervalo de 95%."""
    y = np.asarray(y)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    rng = np.random.default_rng(semente)
    difs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            difs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return float(np.mean(difs)), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def mascaras_janela(anos, nome: str) -> tuple[np.ndarray, np.ndarray]:
    treino, teste = JANELAS[nome]
    anos = np.asarray(anos)
    return np.isin(anos, treino), np.isin(anos, teste)


def avaliar_temporal(fabrica, X: pd.DataFrame, y, anos) -> tuple[dict, dict]:
    """Treina um modelo novo em cada janela e mede no ano seguinte (out-of-time)."""
    y = np.asarray(y)
    metricas, previsoes = {}, {}
    for nome in JANELAS:
        treino, teste = mascaras_janela(anos, nome)
        modelo = fabrica().fit(X.loc[treino], y[treino])
        p = modelo.predict_proba(X.loc[teste])[:, 1]
        metricas[f"auc_{nome}"] = auc(y[teste], p)
        metricas[f"ks_{nome}"] = ks(y[teste], p)
        previsoes[nome] = p
    metricas["auc_medio"] = (metricas["auc_J1"] + metricas["auc_J2"]) / 2
    return metricas, previsoes


def escolher_campeao(candidatos: pd.DataFrame) -> pd.Series:
    """Parte do melhor do nível mais simples; só troca por mais complexo com ganho >= LIMIAR_COMPLEXIDADE."""
    ordenados = candidatos.sort_values(["complexidade", "auc_medio"], ascending=[True, False])
    campeao = ordenados.iloc[0]
    for _, linha in ordenados.iloc[1:].iterrows():
        mais_complexo = linha["complexidade"] > campeao["complexidade"]
        if mais_complexo and linha["auc_medio"] >= campeao["auc_medio"] + LIMIAR_COMPLEXIDADE:
            campeao = linha
    return campeao


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ajustar_platt(y, p) -> tuple[float, float]:
    """Regressão logística (praticamente sem penalidade) do alvo sobre logit(PD): devolve (a, b)."""
    reg = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), np.asarray(y))
    return float(reg.coef_[0, 0]), float(reg.intercept_[0])


def aplicar_platt(p, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def inclinacao_calibracao(y, p) -> float:
    """1 = PDs com a dispersão certa; > 1 = achatadas demais; < 1 = extremas demais."""
    return ajustar_platt(y, p)[0]


def dependencia_parcial(modelo, X: pd.DataFrame, coluna: str, n_pontos: int = 20) -> pd.DataFrame:
    """PD média da carteira quando todos recebem o mesmo valor em `coluna`."""
    serie = X[coluna]
    if pd.api.types.is_numeric_dtype(serie):
        grade = np.nanquantile(serie.to_numpy(dtype=float), np.linspace(0.05, 0.95, n_pontos))
        valores = list(np.unique(grade))
    else:
        valores = sorted(serie.dropna().unique())
    medias = []
    for valor in valores:
        Xv = X.copy()
        Xv[coluna] = valor
        medias.append(float(modelo.predict_proba(Xv)[:, 1].mean()))
    return pd.DataFrame({"valor": valores, "pd_media": medias})
