"""Monta a camada Gold localmente, com o modelo e a política reais, e roda as consultas de exemplo.

Requer as saídas dos notebooks (saidas/modelo_pd.joblib, saidas/tabela_politica.csv e
saidas/politica_base_C.csv). Grava as quatro tabelas em saidas/gold/ e confere que:
- a decisão da tabela Gold é a mesma da submissão oficial da política, proposta a proposta;
- a consulta SQL do motor de decisão reproduz essa decisão;
- o PSI entre as safras de 2023 e 2024 é o mesmo publicado na página.
"""
from datetime import date

import joblib
import numpy as np
import pandas as pd

from autocred import avaliacao, camada_gold, dados, escoragem, politica
from autocred.faixas import atribuir_score

VERSAO_MODELO = "pd_lgbm_monotonico_v1"
VERSAO_POLITICA = "politica_2026_09_v1"
VIGENTE_DESDE = "2026-09-25"  # data da submissão da política
CONSULTAS = dados.RAIZ_PROJETO / "docs" / "arquitetura" / "consultas_gold.sql"


def regras_adotadas(caminho) -> pd.DataFrame:
    """A política decidida no notebook 04, no formato de politica.montar_regras."""
    tabela = pd.read_csv(caminho).set_index("score_1a10")
    return tabela[["aprovar", "taxa_am", "prazo_max", "entrada_min", "aceita_fora_perfil"]]


def conferir(tabelas: dict, submissao: pd.DataFrame, psi_2024: float) -> None:
    decisoes = tabelas["decisoes_credito"].set_index("id_proposta")
    sub = submissao.set_index("id_proposta").loc[decisoes.index]
    assert (decisoes["decisao"] == sub["decisao"]).all(), "decisão diverge da submissão"
    aprovadas = decisoes["decisao"] == "APROVAR"
    assert np.allclose(decisoes.loc[aprovadas, "taxa_am"], sub.loc[aprovadas, "taxa_am"]), "taxa diverge"
    assert (decisoes.loc[aprovadas, "prazo_ofertado_meses"] == sub.loc[aprovadas, "prazo_meses"]).all(), "prazo diverge"
    assert np.allclose(decisoes.loc[aprovadas, "entrada_min_pct"], sub.loc[aprovadas, "pct_entrada_minima"])

    conexao = camada_gold.carregar_sqlite(tabelas)
    via_sql = pd.read_sql_query(camada_gold.ler_consultas(CONSULTAS)["motor_de_decisao_em_lote"], conexao)
    assert (via_sql.set_index("id_proposta")["decisao"] == decisoes["decisao"]).all(), "SQL diverge do Python"

    monitor = tabelas["monitoramento_score"]
    psi_tabela = monitor[monitor["populacao"] == "contratos_2024"]["contribuicao_psi"].sum()
    assert abs(psi_tabela - psi_2024) < 1e-9, "PSI 2023 x 2024 diverge do publicado na página"


def main() -> None:
    hoje = date.today().isoformat()
    saidas = dados.PASTA_SAIDAS
    pacote = joblib.load(saidas / "modelo_pd.joblib")
    modelo, cortes = pacote["modelo_portatil"], pacote["cortes"]
    A, B, C = dados.carregar_base("A"), dados.carregar_base("B"), dados.carregar_base("C")

    base = politica.preparar_base(modelo, C, cortes)
    propostas = camada_gold.propostas_escoradas(base, VERSAO_MODELO, hoje)
    regras = camada_gold.regras_precificacao(regras_adotadas(saidas / "tabela_politica.csv"), VERSAO_POLITICA,
                                             VIGENTE_DESDE)

    scores_a = atribuir_score(escoragem.prever(modelo, A), cortes)
    anos = dados.ano(A).to_numpy()
    populacoes = {f"contratos_{ano}": scores_a[anos == ano] for ano in (2022, 2023, 2024)}
    populacoes["contratos_2025"] = atribuir_score(escoragem.prever(modelo, B), cortes)
    populacoes["propostas_2025_s2"] = propostas["score_1a10"].to_numpy()

    tabelas = {
        "propostas_escoradas": propostas,
        "regras_precificacao": regras,
        "decisoes_credito": camada_gold.decisoes_credito(propostas, regras),
        "monitoramento_score": camada_gold.monitoramento_score(populacoes, "contratos_2023", hoje),
    }
    camada_gold.validar(tabelas)
    psi_2024 = avaliacao.psi(populacoes["contratos_2023"], populacoes["contratos_2024"])
    conferir(tabelas, pd.read_csv(saidas / "politica_base_C.csv"), psi_2024)

    destino = saidas / "gold"
    destino.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_csv(destino / f"{nome}.csv", index=False)
        print(f"gold.{nome}: {len(tabela)} linhas")

    conexao = camada_gold.carregar_sqlite(tabelas)
    with pd.option_context("display.width", 140, "display.max_columns", 12):
        for nome, sql in camada_gold.ler_consultas(CONSULTAS).items():
            print(f"\n-- {nome}")
            print(pd.read_sql_query(sql, conexao).head(10).to_string(index=False))
    print("\nconferências ok: decisões = submissão oficial, SQL = Python, PSI = página")


if __name__ == "__main__":
    main()
