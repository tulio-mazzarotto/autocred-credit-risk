"""Variáveis derivadas e conversão de uma oferta da Base C para o formato da Base A."""
import numpy as np
import pandas as pd

from autocred.dados import CATEGORICAS, COM_AUSENTES, colunas_entrada
from autocred.perda import parcela_price

TAXA_REFERENCIA_AM = 0.01589  # média de taxa_juros_am na Base A

# Colunas que a Base C já traz com o mesmo nome da Base A.
_COMUNS_C = [
    "valor_bem",
    "idade_veiculo_anos",
    "idade_cliente",
    "renda_mensal_declarada",
    "tempo_emprego_meses",
    "score_bureau",
    "qtd_restricoes_ativas",
    "qtd_consultas_bureau_3m",
    *CATEGORICAS,
]


def adicionar_indicadores_ausencia(X: pd.DataFrame) -> pd.DataFrame:
    """Cria ausente_<coluna> (0/1). Não aprende nada com o dado, então vale para qualquer base."""
    X = X.copy()
    for coluna in COM_AUSENTES:
        X[f"ausente_{coluna}"] = X[coluna].isna().astype(int)
    return X


def _vetor(valor, n: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(valor, dtype=float), (n,)).copy()


def oferta_no_formato_a(propostas: pd.DataFrame, taxa_am=TAXA_REFERENCIA_AM, prazo_meses=None, pct_entrada_minima=None) -> pd.DataFrame:
    """Monta as variáveis do modelo para uma oferta (taxa, prazo, entrada mínima) sobre propostas da Base C.

    A parcela e o comprometimento de renda dependem da oferta, por isso são recalculados.
    Sem renda declarada, o comprometimento fica ausente e o Pipeline imputa.
    """
    n = len(propostas)
    bem = propostas["valor_bem"].to_numpy(dtype=float)
    entrada = propostas["valor_entrada_desejada"].to_numpy(dtype=float)
    financiado = propostas["valor_financiado_desejado"].to_numpy(dtype=float)
    ltv = propostas["ltv_desejado"].to_numpy(dtype=float)
    if pct_entrada_minima is not None:
        minima = _vetor(pct_entrada_minima, n)
        sobe = minima > propostas["pct_entrada_desejada"].to_numpy(dtype=float)
        entrada = np.where(sobe, bem * minima, entrada)
        financiado = np.where(sobe, bem - entrada, financiado)
        ltv = np.where(sobe, financiado / bem, ltv)
    if prazo_meses is None:
        prazo = propostas["prazo_desejado_meses"].to_numpy(dtype=float)
    else:
        prazo = _vetor(prazo_meses, n)
    taxa = _vetor(taxa_am, n)
    parcela = parcela_price(financiado, taxa, prazo)

    X = propostas[_COMUNS_C].copy()
    X["valor_entrada"] = entrada
    X["valor_financiado"] = financiado
    X["ltv"] = ltv
    X["prazo_meses"] = prazo.astype(int)
    X["taxa_juros_am"] = taxa
    X["parcela_mensal"] = parcela
    X["comprometimento_renda"] = parcela / X["renda_mensal_declarada"].to_numpy(dtype=float)
    return X[colunas_entrada("completo")]
