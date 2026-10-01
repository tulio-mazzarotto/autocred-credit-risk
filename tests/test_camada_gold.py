import numpy as np
import pandas as pd
import pytest

from autocred import avaliacao, camada_gold, politica
from autocred.dados import RAIZ_PROJETO

CONSULTAS = RAIZ_PROJETO / "docs" / "arquitetura" / "consultas_gold.sql"


@pytest.fixture
def base_escorada():
    """Quatro propostas já escoradas, no formato de politica.preparar_base."""
    return pd.DataFrame({
        "id_proposta": ["P1", "P2", "P3", "P4"],
        "data_proposta": ["2025-07-01", "2025-07-01", "2025-08-01", "2025-08-01"],
        "canal_originacao": ["Digital", "Concessionária", "Digital", "Revenda multimarca"],
        "valor_financiado_desejado": [30000.0, 45000.0, 60000.0, 25000.0],
        "prazo_desejado_meses": [48, 60, 36, 60],
        "pct_entrada_desejada": [0.05, 0.20, 0.30, 0.10],
        "pd_referencia": [0.20, 0.07, 0.04, 0.03],
        "score_1a10": [2, 4, 7, 9],
        "fora_perfil": [False, True, False, True],
    })


@pytest.fixture
def regras():
    # Aprova a partir do score 4, prazo máximo de 48 meses e não aceita quem está fora do perfil.
    taxa = {s: 0.035 - 0.001 * s for s in range(1, 11)}
    return politica.montar_regras(4, taxa, prazo_max=48, entrada_min=0.10, aceita_fora_perfil=False)


@pytest.fixture
def tabelas(base_escorada, regras):
    propostas = camada_gold.propostas_escoradas(base_escorada, "pd_teste_v1", "2025-09-01")
    regras_gold = camada_gold.regras_precificacao(regras, "politica_teste_v1", "2025-09-01")
    return {
        "propostas_escoradas": propostas,
        "regras_precificacao": regras_gold,
        "decisoes_credito": camada_gold.decisoes_credito(propostas, regras_gold),
    }


def test_decisoes_seguem_as_regras_de_preco(tabelas):
    d = tabelas["decisoes_credito"].set_index("id_proposta")
    assert list(d["decisao"]) == ["NEGAR", "NEGAR", "APROVAR", "NEGAR"]
    assert list(d["motivo"]) == ["score abaixo do corte", "fora do perfil", "aprovada", "fora do perfil"]
    assert d.loc["P3", "taxa_am"] == pytest.approx(0.028)
    assert d.loc["P3", "prazo_ofertado_meses"] == 36  # pediu 36, o máximo é 48
    assert d.loc["P3", "entrada_min_pct"] == pytest.approx(0.10)
    assert d.loc[["P1", "P2", "P4"], "taxa_am"].isna().all()  # negadas ficam sem condições


def test_decisoes_batem_com_as_ofertas_da_politica(base_escorada, regras, tabelas):
    ofertas = politica.ofertas(base_escorada, regras).set_index("id_proposta")
    d = tabelas["decisoes_credito"].set_index("id_proposta")
    assert (d["decisao"] == ofertas["decisao"]).all()
    aprovadas = d["decisao"] == "APROVAR"
    assert np.allclose(d.loc[aprovadas, "taxa_am"], ofertas.loc[aprovadas, "taxa_am"])
    assert (d.loc[aprovadas, "prazo_ofertado_meses"] == ofertas.loc[aprovadas, "prazo_meses"]).all()


def test_consulta_do_motor_em_sql_reproduz_a_decisao(tabelas):
    conexao = camada_gold.carregar_sqlite(tabelas)
    sql = camada_gold.ler_consultas(CONSULTAS)["motor_de_decisao_em_lote"]
    via_sql = pd.read_sql_query(sql, conexao).set_index("id_proposta").sort_index()
    via_python = tabelas["decisoes_credito"].set_index("id_proposta").sort_index()
    assert list(via_sql["decisao"]) == list(via_python["decisao"])
    assert list(via_sql["motivo"]) == list(via_python["motivo"])
    assert np.allclose(via_sql["taxa_am"].astype(float), via_python["taxa_am"].astype(float), equal_nan=True)


def test_todas_as_consultas_de_exemplo_rodam(tabelas):
    tabelas = {**tabelas, "monitoramento_score": camada_gold.monitoramento_score(
        {"propostas_2025": np.array([1, 2, 2, 7, 9]), "contratos_2024": np.array([1, 2, 3, 7, 9, 10])},
        "contratos_2024", "2025-09-01")}
    conexao = camada_gold.carregar_sqlite(tabelas)
    consultas = camada_gold.ler_consultas(CONSULTAS)
    assert len(consultas) >= 4
    for nome, sql in consultas.items():
        assert not pd.read_sql_query(sql, conexao).empty, nome


def test_monitoramento_soma_o_psi_da_populacao():
    referencia = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
    atual = np.array([1] * 30 + [2, 3, 4, 5, 6, 7, 8, 9, 10] * 7 + [10] * 7)
    monitor = camada_gold.monitoramento_score({"atual": atual, "ref": referencia}, "ref", "2025-09-01")
    atual_linhas = monitor[monitor["populacao"] == "atual"]
    assert len(atual_linhas) == 10
    assert atual_linhas["contribuicao_psi"].sum() == pytest.approx(avaliacao.psi(referencia, atual))
    assert atual_linhas["pct"].sum() == pytest.approx(1.0)


def test_validar_acusa_tabela_inconsistente(tabelas):
    camada_gold.validar(tabelas)  # as tabelas corretas passam
    quebrada = tabelas["propostas_escoradas"].copy()
    quebrada.loc[0, "pd"] = 1.5
    quebrada.loc[1, "id_proposta"] = "P1"
    with pytest.raises(ValueError, match="pd fora de"):
        camada_gold.validar({**tabelas, "propostas_escoradas": quebrada})
