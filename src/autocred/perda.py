"""Tabela Price, fator de EAD, LGD e perda esperada (parâmetros dados no desafio)."""
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

from autocred.dados import ARQ_PARAMETROS, PASTA_BASES

# Faixas fechadas à esquerda: [0, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, inf).
# Conferido contra os realizados da Base A: é essa convenção que reproduz as tabelas.
CORTES_LTV = [0.60, 0.70, 0.80, 0.90]
# Idade do veículo: 0–2, 3–5, 6–8, 9+ anos.
CORTES_IDADE = [3, 6, 9]
AJUSTE_LGD_AVALISTA = -0.061


class Tabelas(NamedTuple):
    fator_ead: np.ndarray  # linhas = prazos, colunas = faixas de LTV
    lgd: np.ndarray  # linhas = faixas de idade, colunas = faixas de LTV
    prazos: tuple[int, ...]


def parcela_price(valor, taxa_am, prazo_meses):
    """Parcela fixa pela Tabela Price: PMT = PV * i / (1 - (1 + i)^-n)."""
    valor = np.asarray(valor, dtype=float)
    i = np.asarray(taxa_am, dtype=float)
    n = np.asarray(prazo_meses, dtype=float)
    return valor * i / (1 - (1 + i) ** (-n))


def faixa_ltv(ltv) -> np.ndarray:
    return np.searchsorted(CORTES_LTV, np.asarray(ltv, dtype=float), side="right")


def faixa_idade(idade) -> np.ndarray:
    return np.searchsorted(CORTES_IDADE, np.asarray(idade, dtype=float), side="right")


@lru_cache(maxsize=1)
def carregar_tabelas() -> Tabelas:
    arquivo = PASTA_BASES / ARQ_PARAMETROS
    ead = pd.read_excel(arquivo, sheet_name="Fator_EAD", index_col=0)
    lgd_tab = pd.read_excel(arquivo, sheet_name="LGD", index_col=0)
    return Tabelas(
        fator_ead=ead.to_numpy(dtype=float),
        lgd=lgd_tab.to_numpy(dtype=float),
        prazos=tuple(int(p) for p in ead.index),
    )


def fator_ead(prazo_meses, ltv) -> np.ndarray:
    """EAD ÷ valor financiado, por prazo e faixa de LTV."""
    tab = carregar_tabelas()
    prazo = np.asarray(prazo_meses).astype(int)
    if not np.all(np.isin(prazo, tab.prazos)):
        raise ValueError(f"prazo fora da tabela {tab.prazos}: {np.unique(prazo)}")
    linha = np.searchsorted(tab.prazos, prazo)
    return tab.fator_ead[linha, faixa_ltv(ltv)]


def lgd(idade_veiculo_anos, ltv, possui_avalista) -> np.ndarray:
    """LGD da tabela por idade do veículo e LTV, com o ajuste de avalista."""
    tab = carregar_tabelas()
    base = tab.lgd[faixa_idade(idade_veiculo_anos), faixa_ltv(ltv)]
    com_avalista = np.asarray(possui_avalista) == "Sim"
    return base + np.where(com_avalista, AJUSTE_LGD_AVALISTA, 0.0)


def perda_esperada(prob_default, valor_financiado, prazo_meses, ltv, idade_veiculo_anos, possui_avalista) -> pd.DataFrame:
    """Perda esperada = PD × (fator EAD × valor financiado) × LGD."""
    valor = np.asarray(valor_financiado, dtype=float)
    fator = fator_ead(prazo_meses, ltv)
    ead = fator * valor
    perda_lgd = lgd(idade_veiculo_anos, ltv, possui_avalista)
    pe = np.asarray(prob_default, dtype=float) * ead * perda_lgd
    return pd.DataFrame({
        "fator_ead": fator,
        "ead": ead,
        "lgd": perda_lgd,
        "perda_esperada": pe,
        "perda_esperada_pct": pe / valor,
    })
