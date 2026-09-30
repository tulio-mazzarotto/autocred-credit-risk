import numpy as np
import pandas as pd
import pytest

from autocred import perda, politica

CORTES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) / 2
BASE_CEN = politica.CENARIOS[1]


class _PDMetadeDoLTV:
    """Modelo de mentira: PD = LTV / 2."""

    def predict_proba(self, X):
        p = X["ltv"].to_numpy(dtype=float) / 2
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def base(base_c):
    return politica.preparar_base(_PDMetadeDoLTV(), base_c, CORTES)


def test_distribuicao_mes_default():
    d = politica.distribuicao_mes_default()
    assert len(d) == 12
    assert d.sum() == pytest.approx(1.0)


def test_juros_ate_primeira_parcela_e_total():
    pmt = perda.parcela_price(10_000, 0.02, 12)
    assert politica.juros_ate(10_000, 0.02, 12, 1) == pytest.approx(200.0)
    assert politica.juros_ate(10_000, 0.02, 12, 12) == pytest.approx(12 * pmt - 10_000)


def test_juros_esperados_extremos():
    total = politica.juros_ate(10_000, 0.02, 36, 36)
    dist = politica.distribuicao_mes_default()
    com_default = sum(p * politica.juros_ate(10_000, 0.02, 36, m) for m, p in enumerate(dist, start=1))
    assert politica.juros_esperados(10_000, 0.02, 36, 0.0)[0] == pytest.approx(total)
    assert politica.juros_esperados(10_000, 0.02, 36, 1.0)[0] == pytest.approx(com_default)


def test_aceite_na_referencia_e_efeitos():
    ref = politica.TAXA_REFERENCIA_AM
    assert politica.aceite(BASE_CEN, ref, 0, 0) == pytest.approx(0.85)
    assert politica.aceite(BASE_CEN, ref + 0.01, 0, 0) == pytest.approx(0.85 * 0.74)
    assert politica.aceite(BASE_CEN, ref, 0.10, 0) == pytest.approx(0.85 * 0.90)
    assert politica.aceite(BASE_CEN, ref, 0, 12) == pytest.approx(0.85 * 0.90)
    assert politica.aceite(BASE_CEN, 0.010, 0, 0) == pytest.approx(0.85)


def test_pd_ajustada():
    ref = politica.TAXA_REFERENCIA_AM
    assert politica.pd_ajustada(0.10, BASE_CEN, ref) == pytest.approx(0.10)
    assert politica.pd_ajustada(0.10, BASE_CEN, ref + 0.01) == pytest.approx(0.122)
    assert politica.pd_ajustada(0.10, BASE_CEN, ref, fora_perfil=True, margem_fora_perfil=1.5) == pytest.approx(0.15)
    assert politica.pd_ajustada(0.90, BASE_CEN, 0.035) == pytest.approx(0.99)


def test_pd_ajustada_com_estresse_global():
    """O colega estressa a PD de todo mundo; nós, só a de quem está fora do perfil."""
    ref = politica.TAXA_REFERENCIA_AM
    assert politica.pd_ajustada(0.10, BASE_CEN, ref, margem_global=1.2) == pytest.approx(0.12)
    combinado = politica.pd_ajustada(0.10, BASE_CEN, ref, fora_perfil=True, margem_fora_perfil=2.0, margem_global=1.2)
    assert combinado == pytest.approx(0.24)
    assert politica.pd_ajustada(0.50, BASE_CEN, ref, margem_global=2.5) == pytest.approx(0.99)


def test_montar_regras():
    r = politica.montar_regras(5, {s: 0.02 + s / 1000 for s in range(1, 11)}, prazo_max=48)
    assert list(r.index) == list(range(1, 11))
    assert r.loc[4, "aprovar"] == False and r.loc[5, "aprovar"] == True  # noqa: E712
    assert r.loc[10, "taxa_am"] == pytest.approx(0.03)
    assert (r["prazo_max"] == 48).all()


def test_ofertas_aplica_regras(base):
    regras = politica.montar_regras(5, 0.04, prazo_max=36, entrada_min=0.20, aceita_fora_perfil=False)
    of = politica.ofertas(base, regras)
    esperado_aprovar = (base["score_1a10"] >= 5) & ~base["fora_perfil"]
    assert (of["decisao"].eq("APROVAR") == esperado_aprovar).all()
    assert (of["taxa_am"] == 0.035).all()
    assert (of["prazo_meses"] == np.minimum(base["prazo_desejado_meses"], 36)).all()
    np.testing.assert_allclose(of["delta_entrada"], np.maximum(0, 0.20 - base["pct_entrada_desejada"]))


def test_simular_metricas_consistentes(base):
    regras = politica.montar_regras(3, 0.025, prazo_max=48, entrada_min=0.10)
    m, det = politica.simular(_PDMetadeDoLTV(), base, CORTES, regras, BASE_CEN)
    w = det["aceite"]
    assert m["aprovacao"] == pytest.approx(len(det) / len(base))
    assert m["volume"] == pytest.approx((w * det["valor_financiado"]).sum())
    prazo_anos = (w * det["prazo_meses"]).sum() / w.sum() / 12
    roi = ((w * det["juros_esperados"]).sum() - (w * det["perda_esperada"]).sum()) / m["volume"] / prazo_anos
    assert m["roi"] == pytest.approx(roi)
    assert m["ok_volume"] == (m["volume"] >= politica.VOLUME_MIN)


def test_painel_cenarios_ordenados(base):
    regras = politica.montar_regras(3, 0.03)
    p = politica.painel(_PDMetadeDoLTV(), base, CORTES, regras)
    assert list(p.index) == ["brando", "base", "severo"]
    assert p.loc["brando", "volume"] > p.loc["base", "volume"] > p.loc["severo", "volume"]
    assert p.loc["brando", "inadimplencia"] < p.loc["severo", "inadimplencia"]
    assert "ok_limites" in p.columns


def test_submissao_coerente_e_trava(base):
    regras = politica.montar_regras(5, 0.03, prazo_max=48, entrada_min=0.15)
    sub = politica.arquivo_submissao(base, regras)
    assert list(sub.columns) == ["id_proposta", "pd", "score_1a10", "decisao", "taxa_am", "prazo_meses", "pct_entrada_minima"]
    assert sub.loc[sub["decisao"] == "NEGAR", ["taxa_am", "prazo_meses", "pct_entrada_minima"]].isna().all().all()
    politica.validar_coerencia(sub, base, regras, CORTES)
    adulterada = sub.copy()
    linha = adulterada.index[adulterada["decisao"] == "APROVAR"][0]
    adulterada.loc[linha, "taxa_am"] = 0.02
    with pytest.raises(ValueError, match="taxa_am"):
        politica.validar_coerencia(adulterada, base, regras, CORTES)
