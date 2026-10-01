"""Camada Gold: as tabelas prontas para consumo pelo motor de decisão, pelos painéis e pelo monitoramento.

Protótipo local do desenho em docs/arquitetura/camada_gold.md. Em produção, cada função vira um job que
grava uma tabela Delta; aqui, as tabelas são DataFrames e as consultas de exemplo rodam num SQLite em
memória, com o mesmo SQL que rodaria no Databricks.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from autocred.politica import TETO_TAXA_AM

SCORES = range(1, 11)
PISO_PSI = 1e-6  # o mesmo piso de avaliacao.psi, para as contribuições somarem o PSI


def propostas_escoradas(base: pd.DataFrame, versao_modelo: str, data_processamento: str) -> pd.DataFrame:
    """gold.propostas_escoradas: uma linha por proposta, com o pedido do cliente, a PD e o score."""
    return pd.DataFrame({
        "id_proposta": base["id_proposta"].to_numpy(),
        "data_proposta": base["data_proposta"].to_numpy(),
        "canal_originacao": base["canal_originacao"].to_numpy(),
        "valor_financiado_desejado": base["valor_financiado_desejado"].to_numpy(dtype=float),
        "prazo_desejado_meses": base["prazo_desejado_meses"].to_numpy(dtype=int),
        "pct_entrada_desejada": base["pct_entrada_desejada"].to_numpy(dtype=float),
        "pd": base["pd_referencia"].to_numpy(dtype=float),
        "score_1a10": base["score_1a10"].to_numpy(dtype=int),
        "fora_perfil": base["fora_perfil"].to_numpy(dtype=bool),
        "versao_modelo": versao_modelo,
        "data_processamento": data_processamento,
    })


def regras_precificacao(regras: pd.DataFrame, versao_politica: str, vigente_desde: str) -> pd.DataFrame:
    """gold.regras_precificacao: a política por score, versionada, com vigência (histórico tipo SCD2)."""
    r = regras.reset_index()
    return pd.DataFrame({
        "versao_politica": versao_politica,
        "score_1a10": r["score_1a10"].to_numpy(dtype=int),
        "aprovar": r["aprovar"].to_numpy(dtype=bool),
        "taxa_am": np.minimum(r["taxa_am"].to_numpy(dtype=float), TETO_TAXA_AM),
        "prazo_max_meses": r["prazo_max"].to_numpy(dtype=int),
        "entrada_min_pct": r["entrada_min"].to_numpy(dtype=float),
        "aceita_fora_perfil": r["aceita_fora_perfil"].to_numpy(dtype=bool),
        "vigente_desde": vigente_desde,
        "vigente_ate": None,  # nulo = versão em vigor
    })


def decisoes_credito(propostas: pd.DataFrame, regras: pd.DataFrame) -> pd.DataFrame:
    """gold.decisoes_credito: a decisão e as condições ofertadas a cada proposta, com o motivo.

    É a mesma lógica da consulta "motor_de_decisao_em_lote" em docs/arquitetura/consultas_gold.sql.
    """
    vigentes = regras[regras["vigente_ate"].isna()]
    if vigentes["score_1a10"].duplicated().any():
        raise ValueError("Há mais de uma versão de política em vigor para o mesmo score.")
    d = propostas.merge(vigentes, on="score_1a10", how="left", validate="many_to_one")
    aprova_score = d["aprovar"].to_numpy(dtype=bool)
    barra_perfil = d["fora_perfil"].to_numpy(dtype=bool) & ~d["aceita_fora_perfil"].to_numpy(dtype=bool)
    aprovada = aprova_score & ~barra_perfil
    prazo = np.minimum(d["prazo_desejado_meses"].to_numpy(), d["prazo_max_meses"].to_numpy())
    return pd.DataFrame({
        "id_proposta": d["id_proposta"].to_numpy(),
        "data_proposta": d["data_proposta"].to_numpy(),
        "canal_originacao": d["canal_originacao"].to_numpy(),
        "score_1a10": d["score_1a10"].to_numpy(),
        "fora_perfil": d["fora_perfil"].to_numpy(dtype=bool),
        "decisao": np.where(aprovada, "APROVAR", "NEGAR"),
        "motivo": np.select([~aprova_score, barra_perfil], ["score abaixo do corte", "fora do perfil"], "aprovada"),
        "taxa_am": np.where(aprovada, d["taxa_am"].to_numpy(dtype=float), np.nan),
        "prazo_ofertado_meses": pd.array(np.where(aprovada, prazo, np.nan), dtype="Int64"),
        "entrada_min_pct": np.where(aprovada, d["entrada_min_pct"].to_numpy(dtype=float), np.nan),
        "versao_modelo": d["versao_modelo"].to_numpy(),
        "versao_politica": d["versao_politica"].to_numpy(),
    })


def _distribuicao(scores) -> np.ndarray:
    scores = np.asarray(scores)
    return np.array([np.mean(scores == s) for s in SCORES])


def monitoramento_score(scores_por_populacao: dict, referencia: str, data_processamento: str) -> pd.DataFrame:
    """gold.monitoramento_score: distribuição por score de cada população e a contribuição de cada faixa
    para o PSI contra a população de referência. Somando a contribuição por população, sai o PSI."""
    ref = _distribuicao(scores_por_populacao[referencia])
    linhas = []
    for populacao, scores in scores_por_populacao.items():
        atual = _distribuicao(scores)
        a, r = np.clip(atual, PISO_PSI, None), np.clip(ref, PISO_PSI, None)
        contribuicao = (a - r) * np.log(a / r)
        for i, score in enumerate(SCORES):
            linhas.append({
                "populacao": populacao, "referencia": referencia, "score_1a10": score,
                "qtd": int(np.sum(np.asarray(scores) == score)), "pct": float(atual[i]),
                "pct_referencia": float(ref[i]), "contribuicao_psi": float(contribuicao[i]),
                "data_processamento": data_processamento,
            })
    return pd.DataFrame(linhas)


def validar(tabelas: dict) -> None:
    """Checagens de qualidade que, em produção, bloqueiam a publicação da tabela (expectations)."""
    problemas = []
    p = tabelas.get("propostas_escoradas")
    if p is not None:
        if p["id_proposta"].duplicated().any():
            problemas.append("propostas_escoradas: id_proposta repetido")
        if not p["pd"].between(0, 1).all():
            problemas.append("propostas_escoradas: pd fora de [0, 1]")
        if not p["score_1a10"].isin(SCORES).all():
            problemas.append("propostas_escoradas: score fora de 1 a 10")
    r = tabelas.get("regras_precificacao")
    if r is not None:
        if r[r["vigente_ate"].isna()]["score_1a10"].duplicated().any():
            problemas.append("regras_precificacao: mais de uma regra em vigor por score")
        if (r["taxa_am"] > TETO_TAXA_AM).any():
            problemas.append("regras_precificacao: taxa acima do teto")
        if not r["entrada_min_pct"].between(0, 1).all():
            problemas.append("regras_precificacao: entrada mínima fora de [0, 1]")
    d = tabelas.get("decisoes_credito")
    if d is not None:
        if p is not None and set(d["id_proposta"]) != set(p["id_proposta"]):
            problemas.append("decisoes_credito: não cobre exatamente as propostas escoradas")
        aprovadas = d["decisao"] == "APROVAR"
        if d.loc[aprovadas, "taxa_am"].isna().any() or d.loc[~aprovadas, "taxa_am"].notna().any():
            problemas.append("decisoes_credito: condições incoerentes com a decisão")
    m = tabelas.get("monitoramento_score")
    if m is not None and not np.allclose(m.groupby("populacao")["pct"].sum(), 1.0):
        problemas.append("monitoramento_score: distribuição que não soma 100%")
    if problemas:
        raise ValueError("; ".join(problemas))


def carregar_sqlite(tabelas: dict) -> sqlite3.Connection:
    """Carrega as tabelas num SQLite em memória, no esquema "gold", para rodar o SQL de exemplo."""
    conexao = sqlite3.connect(":memory:")
    conexao.execute("ATTACH DATABASE ':memory:' AS gold")
    for nome, tabela in tabelas.items():
        tabela.to_sql(nome, conexao, index=False)
        conexao.execute(f"CREATE TABLE gold.{nome} AS SELECT * FROM main.{nome}")
        conexao.execute(f"DROP TABLE main.{nome}")
    return conexao


def ler_consultas(caminho: Path) -> dict:
    """Lê o arquivo .sql de exemplos: cada consulta começa com uma linha "-- consulta: <nome>"."""
    texto = Path(caminho).read_text(encoding="utf-8")
    blocos = re.split(r"^-- consulta: (\w+)\s*$", texto, flags=re.MULTILINE)[1:]
    return {nome: sql.strip().rstrip(";") for nome, sql in zip(blocos[::2], blocos[1::2])}
