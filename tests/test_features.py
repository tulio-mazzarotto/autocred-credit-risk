import numpy as np
import pandas as pd
import pytest

from autocred import dados, features, perda


def test_indicadores_de_ausencia_nao_alteram_original():
    X = pd.DataFrame({
        "score_bureau": [700.0, np.nan],
        "renda_mensal_declarada": [np.nan, 5000.0],
        "tempo_emprego_meses": [10.0, 20.0],
    })
    r = features.adicionar_indicadores_ausencia(X)
    assert r["ausente_score_bureau"].tolist() == [0, 1]
    assert r["ausente_renda_mensal_declarada"].tolist() == [1, 0]
    assert r["ausente_tempo_emprego_meses"].tolist() == [0, 0]
    assert "ausente_score_bureau" not in X.columns


def test_oferta_sem_mudancas_reproduz_pedido(base_c):
    X = features.oferta_no_formato_a(base_c)
    assert list(X.columns) == dados.colunas_entrada("completo")
    assert X.index.equals(base_c.index)
    np.testing.assert_allclose(X["valor_financiado"], base_c["valor_financiado_desejado"])
    np.testing.assert_allclose(X["ltv"], base_c["ltv_desejado"])
    assert (X["prazo_meses"] == base_c["prazo_desejado_meses"]).all()
    assert (X["taxa_juros_am"] == features.TAXA_REFERENCIA_AM).all()
    esperada = perda.parcela_price(base_c["valor_financiado_desejado"], 0.01589, base_c["prazo_desejado_meses"])
    np.testing.assert_allclose(X["parcela_mensal"], esperada)
    assert (X["comprometimento_renda"].isna() == base_c["renda_mensal_declarada"].isna()).all()


def test_entrada_minima_so_afeta_quem_pediu_menos(base_c):
    X = features.oferta_no_formato_a(base_c, pct_entrada_minima=0.30)
    afetados = base_c["pct_entrada_desejada"] < 0.30
    np.testing.assert_allclose(X.loc[afetados, "ltv"], 0.70)
    np.testing.assert_allclose(X.loc[afetados, "valor_financiado"], base_c.loc[afetados, "valor_bem"] * 0.70)
    np.testing.assert_allclose(X.loc[~afetados, "ltv"], base_c.loc[~afetados, "ltv_desejado"])
    assert afetados.any() and (~afetados).any()


def test_prazo_ofertado_recalcula_parcela(base_c):
    X = features.oferta_no_formato_a(base_c, prazo_meses=36)
    assert (X["prazo_meses"] == 36).all()
    esperada = perda.parcela_price(base_c["valor_financiado_desejado"], 0.01589, 36)
    np.testing.assert_allclose(X["parcela_mensal"], esperada)


def test_taxa_por_proposta(base_c):
    taxas = np.where(base_c["score_bureau"].fillna(0) > 600, 0.02, 0.03)
    X = features.oferta_no_formato_a(base_c, taxa_am=taxas)
    np.testing.assert_allclose(X["taxa_juros_am"], taxas)
