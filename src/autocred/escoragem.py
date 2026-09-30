"""Escoragem: PD, score 1–10 e perda esperada; ofertas da Base C; submissão da Base B."""
from pathlib import Path

import numpy as np
import pandas as pd

from autocred.dados import colunas_entrada
from autocred.faixas import atribuir_score
from autocred.features import TAXA_REFERENCIA_AM, oferta_no_formato_a
from autocred.perda import perda_esperada

# Limites da política antiga, observados na Base A (o modelo nunca viu nada além deles).
LIMITE_SCORE_BUREAU = 460
LIMITE_RESTRICOES = 2
LIMITE_LTV = 0.95


def prever(modelo, X: pd.DataFrame, conjunto: str = "portatil") -> np.ndarray:
    return modelo.predict_proba(X[colunas_entrada(conjunto)])[:, 1]


def escorar(modelo, X: pd.DataFrame, cortes, conjunto: str = "portatil") -> pd.DataFrame:
    """PD, score e perda esperada para linhas no formato da Base A."""
    p = prever(modelo, X, conjunto)
    perda = perda_esperada(
        p, X["valor_financiado"], X["prazo_meses"], X["ltv"], X["idade_veiculo_anos"], X["possui_avalista"]
    )
    perda.index = X.index
    saida = pd.DataFrame({"pd": p, "score_1a10": atribuir_score(p, cortes)}, index=X.index)
    return pd.concat([saida, perda], axis=1)


def fora_perfil_historico(X: pd.DataFrame) -> pd.Series:
    """True quando a proposta está fora do que a política antiga aprovava. Bureau ausente não conta."""
    return (
        (X["score_bureau"] < LIMITE_SCORE_BUREAU)
        | (X["qtd_restricoes_ativas"] > LIMITE_RESTRICOES)
        | (X["ltv"] > LIMITE_LTV)
    )


def prever_pd(modelo, propostas: pd.DataFrame, cortes, taxa_am=TAXA_REFERENCIA_AM, prazo_meses=None,
              pct_entrada_minima=None, conjunto: str = "portatil") -> pd.DataFrame:
    """PD e perda esperada de propostas da Base C sob uma oferta (taxa, prazo, entrada mínima).

    Atenção: o modelo não enxerga a seleção adversa da taxa, e a PD de quem está fora
    do perfil histórico é extrapolação.
    """
    X = oferta_no_formato_a(propostas, taxa_am, prazo_meses, pct_entrada_minima)
    r = escorar(modelo, X, cortes, conjunto)
    r.insert(0, "id_proposta", propostas["id_proposta"].to_numpy())
    r["ltv_ofertado"] = X["ltv"].to_numpy()
    r["prazo_meses"] = X["prazo_meses"].to_numpy()
    r["taxa_am"] = X["taxa_juros_am"].to_numpy()
    r["fora_perfil_historico"] = fora_perfil_historico(X).to_numpy()
    return r


def validar_submissao_b(submissao: pd.DataFrame, base_b: pd.DataFrame) -> None:
    if list(submissao.columns) != ["id_contrato", "pd"]:
        raise ValueError(f"colunas devem ser ['id_contrato', 'pd'], vieram {list(submissao.columns)}")
    if len(submissao) != len(base_b):
        raise ValueError(f"esperava {len(base_b)} linhas, vieram {len(submissao)}")
    if not np.array_equal(submissao["id_contrato"].to_numpy(), base_b["id_contrato"].to_numpy()):
        raise ValueError("ids diferentes ou fora da ordem da Base B")
    if submissao["pd"].isna().any():
        raise ValueError("há PD vazia")
    if not ((submissao["pd"] > 0) & (submissao["pd"] < 1)).all():
        raise ValueError("toda PD deve estar entre 0 e 1 (exclusive)")


def salvar_submissao_b(ids, prob_default, base_b: pd.DataFrame, caminho) -> pd.DataFrame:
    """Grava com 6 casas (4 casas criariam empates desnecessários), relê e valida."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id_contrato": np.asarray(ids), "pd": np.asarray(prob_default, dtype=float)})
    sub.to_csv(caminho, index=False, float_format="%.6f")
    relida = pd.read_csv(caminho)
    validar_submissao_b(relida, base_b)
    return relida
