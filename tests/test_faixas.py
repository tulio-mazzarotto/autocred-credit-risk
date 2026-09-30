import numpy as np
import pandas as pd
import pytest

from autocred import faixas

CORTES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_atribuir_score_nas_fronteiras():
    assert faixas.atribuir_score([0.05, 0.10, 0.15, 0.95], CORTES).tolist() == [10, 9, 9, 1]


def test_maior_pd_nunca_tem_score_maior():
    p = np.sort(np.random.default_rng(0).uniform(size=500))
    score = faixas.atribuir_score(p, faixas.cortes_decis(p))
    assert np.all(np.diff(score) <= 0)


def test_decis_dividem_em_10_grupos_parecidos():
    p = np.random.default_rng(1).uniform(size=1000)
    contagem = pd.Series(faixas.atribuir_score(p, faixas.cortes_decis(p))).value_counts()
    assert sorted(contagem.index) == list(range(1, 11))
    assert contagem.between(95, 105).all()


def test_cortes_geometricos():
    c = faixas.cortes_geometricos(0.01, 0.5)
    assert len(c) == 9
    razoes = c[1:] / c[:-1]
    np.testing.assert_allclose(razoes, razoes[0])
    assert 0.01 < c[0] and c[-1] < 0.5


def test_inadimplencia_por_decil():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.9).astype(int)  # só o decil de maior PD é inadimplente
    r = faixas.inadimplencia_por_decil(y, p)
    assert list(r.index) == list(range(1, 11))
    assert r.loc[1] == pytest.approx(1.0)
    assert r.loc[2:].sum() == pytest.approx(0.0)


def test_tabela_faixas():
    p = np.linspace(0.01, 0.30, 100)
    escorada = pd.DataFrame({
        "pd": p,
        "score_1a10": faixas.atribuir_score(p, faixas.cortes_decis(p)),
        "fator_ead": 1.02,
        "lgd": 0.7,
        "perda_esperada_pct": p * 1.02 * 0.7,
    })
    t = faixas.tabela_faixas(escorada)
    assert list(t.index) == list(range(1, 11))
    assert t["qtd"].sum() == 100
    assert t["pd_media"].is_monotonic_decreasing
    assert list(t.columns) == [
        "qtd", "pd_min", "pd_max", "pd_media", "fator_ead_medio", "lgd_media", "perda_esperada_pct_media",
    ]
