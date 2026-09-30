"""Simulador de política de crédito na Base C: ofertas, reação do cliente e resultado da carteira."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from autocred import escoragem
from autocred.dados import ARQ_PARAMETROS, PASTA_BASES
from autocred.faixas import atribuir_score
from autocred.features import TAXA_REFERENCIA_AM, oferta_no_formato_a
from autocred.perda import parcela_price

TETO_TAXA_AM = 0.035
APROVACAO_MIN = 0.35
INADIMPLENCIA_MAX = 0.08
VOLUME_MIN = 40_000_000.0
ROI_ALVO = 0.15


@dataclass(frozen=True)
class Cenario:
    """Premissas de reação do cliente. Os efeitos são por +1 p.p. a.m. de taxa acima da referência,
    por +10 p.p. de entrada exigida além da desejada e por 12 meses de prazo cortados."""

    nome: str
    aceite_base: float
    fator_taxa: float
    queda_entrada: float
    queda_prazo: float
    selecao_adversa: float


CENARIOS = (
    Cenario("brando", 0.90, 0.86, 0.05, 0.05, 0.10),
    Cenario("base", 0.85, 0.74, 0.10, 0.10, 0.22),
    Cenario("severo", 0.80, 0.61, 0.15, 0.15, 0.42),
)


@lru_cache(maxsize=1)
def distribuicao_mes_default() -> np.ndarray:
    """Probabilidade de o calote cair em cada mês 1–12 (aba do arquivo de parâmetros)."""
    tabela = pd.read_excel(PASTA_BASES / ARQ_PARAMETROS, sheet_name="Distribuicao_Mes_Default")
    p = tabela.iloc[:, 1].to_numpy(dtype=float)
    return p / p.sum()


def _delta_taxa_pp(taxa_am) -> np.ndarray:
    return np.maximum(0.0, (np.asarray(taxa_am, dtype=float) - TAXA_REFERENCIA_AM) * 100)


def juros_ate(valor, taxa_am, prazo_meses, meses):
    """Juros pagos nas primeiras `meses` parcelas da Tabela Price: parcelas pagas − principal amortizado."""
    pv = np.asarray(valor, dtype=float)
    i = np.asarray(taxa_am, dtype=float)
    n = np.asarray(prazo_meses, dtype=float)
    m = np.asarray(meses, dtype=float)
    pmt = parcela_price(pv, i, n)
    saldo = pv * (1 + i) ** m - pmt * ((1 + i) ** m - 1) / i
    return m * pmt - (pv - saldo)


def juros_esperados(valor, taxa_am, prazo_meses, prob_default) -> np.ndarray:
    """(1 − PD) × juros do prazo inteiro + PD × juros esperados até o mês do calote."""
    pv = np.atleast_1d(np.asarray(valor, dtype=float))
    i = np.atleast_1d(np.asarray(taxa_am, dtype=float))
    n = np.atleast_1d(np.asarray(prazo_meses, dtype=float))
    p = np.atleast_1d(np.asarray(prob_default, dtype=float))
    dist = distribuicao_mes_default()
    meses = np.arange(1, len(dist) + 1, dtype=float)
    com_default = juros_ate(pv[:, None], i[:, None], n[:, None], meses[None, :]) @ dist
    sem_default = juros_ate(pv, i, n, n)
    return (1 - p) * sem_default + p * com_default


def aceite(cenario: Cenario, taxa_am, delta_entrada, meses_cortados) -> np.ndarray:
    """Chance de o cliente aceitar a oferta. Taxa abaixo da referência não aumenta o aceite."""
    a = cenario.aceite_base * cenario.fator_taxa ** _delta_taxa_pp(taxa_am)
    a = a * np.maximum(0.0, 1 - cenario.queda_entrada * np.asarray(delta_entrada, dtype=float) / 0.10)
    return a * np.maximum(0.0, 1 - cenario.queda_prazo * np.asarray(meses_cortados, dtype=float) / 12)


def pd_ajustada(prob_default, cenario: Cenario, taxa_am, fora_perfil=False, margem_fora_perfil: float = 1.0,
                margem_global: float = 1.0) -> np.ndarray:
    """PD do modelo × seleção adversa da taxa × margens de estresse (teto de 0,99).

    `margem_global` testa o nível da PD estar errado para todo mundo; `margem_fora_perfil`
    testa só o grupo que o modelo nunca viu. As duas se multiplicam.
    """
    p = np.asarray(prob_default, dtype=float) * (1 + cenario.selecao_adversa) ** _delta_taxa_pp(taxa_am)
    p = p * margem_global * np.where(np.asarray(fora_perfil), margem_fora_perfil, 1.0)
    return np.minimum(p, 0.99)


def _por_score(valor) -> list:
    if isinstance(valor, dict):
        return [valor[s] for s in range(1, 11)]
    return [valor] * 10


def montar_regras(score_minimo: int, taxa_am, prazo_max=60, entrada_min=0.0, aceita_fora_perfil=True) -> pd.DataFrame:
    """Tabela de regras por score. Cada parâmetro aceita um valor único ou um dict {score: valor}."""
    scores = range(1, 11)
    return pd.DataFrame(
        {
            "aprovar": [s >= score_minimo for s in scores],
            "taxa_am": _por_score(taxa_am),
            "prazo_max": _por_score(prazo_max),
            "entrada_min": _por_score(entrada_min),
            "aceita_fora_perfil": _por_score(aceita_fora_perfil),
        },
        index=pd.Index(scores, name="score_1a10"),
    )


def preparar_base(modelo, propostas: pd.DataFrame, cortes) -> pd.DataFrame:
    """Score e perfil de cada proposta nas condições pedidas (a oferta não muda a faixa)."""
    referencia = escoragem.prever_pd(modelo, propostas, cortes)
    base = propostas.copy()
    base["pd_referencia"] = referencia["pd"].to_numpy()
    base["score_1a10"] = referencia["score_1a10"].to_numpy()
    base["fora_perfil"] = referencia["fora_perfil_historico"].to_numpy()
    return base


def ofertas(base: pd.DataFrame, regras: pd.DataFrame) -> pd.DataFrame:
    """Aplica a tabela de regras a cada proposta: decisão e condições ofertadas."""
    r = regras.loc[base["score_1a10"].to_numpy()]
    fora = base["fora_perfil"].to_numpy(dtype=bool)
    aprovar = r["aprovar"].to_numpy(dtype=bool) & (~fora | r["aceita_fora_perfil"].to_numpy(dtype=bool))
    prazo_desejado = base["prazo_desejado_meses"].to_numpy()
    prazo = np.minimum(prazo_desejado, r["prazo_max"].to_numpy())
    entrada_min = r["entrada_min"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "id_proposta": base["id_proposta"].to_numpy(),
            "score_1a10": base["score_1a10"].to_numpy(),
            "fora_perfil": fora,
            "decisao": np.where(aprovar, "APROVAR", "NEGAR"),
            "taxa_am": np.minimum(r["taxa_am"].to_numpy(dtype=float), TETO_TAXA_AM),
            "prazo_meses": prazo.astype(int),
            "pct_entrada_minima": entrada_min,
            "delta_entrada": np.maximum(0.0, entrada_min - base["pct_entrada_desejada"].to_numpy(dtype=float)),
            "meses_cortados": prazo_desejado - prazo,
        },
        index=base.index,
    )


def resumir(detalhe: pd.DataFrame, n_propostas: int) -> dict:
    """Métricas da carteira, ponderadas pela chance de aceite de cada aprovado."""
    w = detalhe["aceite"].to_numpy()
    contratos = float(w.sum())
    aprovacao = len(detalhe) / n_propostas
    if contratos == 0:
        return {"aprovacao": aprovacao, "contratos": 0.0, "volume": 0.0, "inadimplencia": 0.0,
                "prazo_medio_anos": 0.0, "roi": float("nan"),
                "ok_aprovacao": aprovacao >= APROVACAO_MIN, "ok_inadimplencia": True, "ok_volume": False}
    volume = float((w * detalhe["valor_financiado"]).sum())
    inadimplencia = float((w * detalhe["pd_simulada"]).sum() / contratos)
    prazo_anos = float((w * detalhe["prazo_meses"]).sum() / contratos / 12)
    lucro = float((w * detalhe["juros_esperados"]).sum() - (w * detalhe["perda_esperada"]).sum())
    return {
        "aprovacao": aprovacao,
        "contratos": contratos,
        "volume": volume,
        "inadimplencia": inadimplencia,
        "prazo_medio_anos": prazo_anos,
        "roi": lucro / volume / prazo_anos,
        "ok_aprovacao": bool(aprovacao >= APROVACAO_MIN),
        "ok_inadimplencia": bool(inadimplencia <= INADIMPLENCIA_MAX),
        "ok_volume": bool(volume >= VOLUME_MIN),
    }


def simular(modelo, base: pd.DataFrame, cortes, regras: pd.DataFrame, cenario: Cenario,
            margem_fora_perfil: float = 1.0, margem_global: float = 1.0) -> tuple[dict, pd.DataFrame]:
    """Resultado esperado da carteira sob uma política e um cenário de reação do cliente."""
    of = ofertas(base, regras)
    aprovado = (of["decisao"] == "APROVAR").to_numpy()
    b = base.loc[aprovado]
    o = of.loc[aprovado]
    X = oferta_no_formato_a(b, o["taxa_am"].to_numpy(), o["prazo_meses"].to_numpy(), o["pct_entrada_minima"].to_numpy())
    esc = escoragem.escorar(modelo, X, cortes)
    taxa = o["taxa_am"].to_numpy()
    pd_sim = pd_ajustada(
        esc["pd"].to_numpy(), cenario, taxa, o["fora_perfil"].to_numpy(), margem_fora_perfil, margem_global
    )
    detalhe = pd.DataFrame(
        {
            "id_proposta": o["id_proposta"].to_numpy(),
            "score_1a10": o["score_1a10"].to_numpy(),
            "aceite": aceite(cenario, taxa, o["delta_entrada"].to_numpy(), o["meses_cortados"].to_numpy()),
            "pd_simulada": pd_sim,
            "valor_financiado": X["valor_financiado"].to_numpy(dtype=float),
            "prazo_meses": X["prazo_meses"].to_numpy(dtype=float),
            "taxa_am": taxa,
            "juros_esperados": juros_esperados(X["valor_financiado"], taxa, X["prazo_meses"], pd_sim),
            "perda_esperada": pd_sim * esc["ead"].to_numpy() * esc["lgd"].to_numpy(),
        },
        index=b.index,
    )
    return resumir(detalhe, len(base)), detalhe


def painel(modelo, base: pd.DataFrame, cortes, regras: pd.DataFrame, margem_fora_perfil: float = 1.0,
           margem_global: float = 1.0) -> pd.DataFrame:
    """Uma linha por cenário, com as métricas e se cada limite do conselho foi cumprido."""
    linhas = []
    for cenario in CENARIOS:
        metricas, _ = simular(modelo, base, cortes, regras, cenario, margem_fora_perfil, margem_global)
        linhas.append({"cenario": cenario.nome, **metricas})
    tabela = pd.DataFrame(linhas).set_index("cenario")
    tabela["ok_limites"] = tabela[["ok_aprovacao", "ok_inadimplencia", "ok_volume"]].all(axis=1)
    return tabela


def arquivo_submissao(base: pd.DataFrame, regras: pd.DataFrame) -> pd.DataFrame:
    """Formato do CSV de exemplo da política. Linhas NEGAR ficam sem taxa, prazo e entrada."""
    of = ofertas(base, regras)
    negar = (of["decisao"] == "NEGAR").to_numpy()
    sub = pd.DataFrame({
        "id_proposta": of["id_proposta"].to_numpy(),
        "pd": np.round(base["pd_referencia"].to_numpy(dtype=float), 6),
        "score_1a10": of["score_1a10"].to_numpy(),
        "decisao": of["decisao"].to_numpy(),
        "taxa_am": np.where(negar, np.nan, of["taxa_am"].to_numpy()),
        "prazo_meses": pd.array(np.where(negar, np.nan, of["prazo_meses"].to_numpy(dtype=float)), dtype="Int64"),
        "pct_entrada_minima": np.where(negar, np.nan, of["pct_entrada_minima"].to_numpy()),
    })
    return sub


def validar_coerencia(submissao: pd.DataFrame, base: pd.DataFrame, regras: pd.DataFrame, cortes) -> None:
    """Trava: cada linha submetida tem de seguir a regra da sua faixa (vale 10 pontos na avaliação)."""
    esperado = arquivo_submissao(base, regras)
    if len(submissao) != len(esperado):
        raise ValueError(f"esperava {len(esperado)} linhas, vieram {len(submissao)}")
    if not np.array_equal(submissao["id_proposta"].to_numpy(), esperado["id_proposta"].to_numpy()):
        raise ValueError("ids diferentes ou fora da ordem da Base C")
    if not np.array_equal(atribuir_score(submissao["pd"], cortes), submissao["score_1a10"].to_numpy()):
        raise ValueError("score_1a10 incoerente com a PD e os cortes")
    for coluna in ["decisao", "taxa_am", "prazo_meses", "pct_entrada_minima"]:
        a, b = submissao[coluna], esperado[coluna]
        iguais = (a.isna() & b.isna()) | (a == b)
        if not iguais.fillna(False).all():
            raise ValueError(f"coluna {coluna} diverge da tabela de regras")
