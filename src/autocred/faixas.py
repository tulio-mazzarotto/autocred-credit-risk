"""Faixas de score 1 (pior) a 10 (melhor) e resumo por faixa."""
import numpy as np
import pandas as pd


def cortes_decis(pd_referencia) -> np.ndarray:
    """9 cortes internos: cada faixa fica com ~10% da população de referência."""
    return np.quantile(np.asarray(pd_referencia, dtype=float), np.linspace(0.1, 0.9, 9))


def cortes_geometricos(pd_min: float, pd_max: float, n_faixas: int = 10) -> np.ndarray:
    """Alternativa: cortes em escala geométrica (cada faixa multiplica a PD pelo mesmo fator)."""
    return np.geomspace(pd_min, pd_max, n_faixas + 1)[1:-1]


def atribuir_score(prob_default, cortes) -> np.ndarray:
    """Score 1 = maior PD, 10 = menor PD. Empate com um corte vai para a faixa de maior risco."""
    posicao = np.searchsorted(np.asarray(cortes, dtype=float), np.asarray(prob_default, dtype=float), side="right")
    return 10 - posicao


def inadimplencia_por_decil(y, prob_default) -> pd.Series:
    """Inadimplência observada em cada decil da PD (decil 1 = PDs mais altas)."""
    p = np.asarray(prob_default, dtype=float)
    score = atribuir_score(p, cortes_decis(p))
    return pd.Series(np.asarray(y)).groupby(score).mean().rename("inadimplencia_observada_J2")


def tabela_faixas(escorada: pd.DataFrame) -> pd.DataFrame:
    grupos = escorada.groupby("score_1a10")
    tabela = pd.DataFrame({
        "qtd": grupos.size(),
        "pd_min": grupos["pd"].min(),
        "pd_max": grupos["pd"].max(),
        "pd_media": grupos["pd"].mean(),
        "fator_ead_medio": grupos["fator_ead"].mean(),
        "lgd_media": grupos["lgd"].mean(),
        "perda_esperada_pct_media": grupos["perda_esperada_pct"].mean(),
    })
    return tabela.reindex(range(1, 11)).rename_axis("score_1a10")
