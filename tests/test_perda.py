import numpy as np
import pandas as pd
import pytest

from autocred import perda


def test_price_reproduz_parcelas_da_base_a(base_a):
    calculada = perda.parcela_price(base_a["valor_financiado"], base_a["taxa_juros_am"], base_a["prazo_meses"])
    assert np.max(np.abs(calculada - base_a["parcela_mensal"])) <= 0.05


def test_faixa_ltv_fechada_a_esquerda():
    ltv = [0.59, 0.60, 0.6999, 0.70, 0.80, 0.90, 0.95]
    assert list(perda.faixa_ltv(ltv)) == [0, 1, 1, 2, 3, 4, 4]


def test_faixa_idade():
    idades = [0, 2, 3, 5, 6, 8, 9, 12]
    assert list(perda.faixa_idade(idades)) == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize(
    ("prazo", "ltv", "esperado"),
    [(24, 0.50, 0.980), (36, 0.7999, 1.020), (36, 0.80, 1.015), (36, 0.85, 1.015), (60, 0.95, 1.042)],
)
def test_fator_ead(prazo, ltv, esperado):
    assert perda.fator_ead(prazo, ltv) == pytest.approx(esperado)


def test_fator_ead_prazo_invalido():
    with pytest.raises(ValueError, match="prazo"):
        perda.fator_ead([30], [0.5])


@pytest.mark.parametrize(
    ("idade", "ltv", "avalista", "esperado"),
    [
        (2, 0.59, "Não", 0.412),
        (3, 0.60, "Não", 0.622),
        (0, 0.85, "Não", 0.652),
        (0, 0.85, "Sim", 0.652 - 0.061),
        (9, 0.95, "Não", 0.908),
    ],
)
def test_lgd(idade, ltv, avalista, esperado):
    assert perda.lgd(idade, ltv, avalista) == pytest.approx(esperado)


def test_perda_esperada_exemplo_manual():
    r = perda.perda_esperada([0.10], [10_000.0], [36], [0.85], [0], ["Não"])
    linha = r.iloc[0]
    assert linha["fator_ead"] == pytest.approx(1.015)
    assert linha["ead"] == pytest.approx(10_150.0)
    assert linha["lgd"] == pytest.approx(0.652)
    assert linha["perda_esperada"] == pytest.approx(0.10 * 10_150.0 * 0.652)
    assert linha["perda_esperada_pct"] == pytest.approx(0.10 * 1.015 * 0.652)


def test_tabelas_batem_com_realizados_da_base_a(base_a):
    """Conferência pedida no enunciado: os parâmetros saem dos realizados da Base A."""
    tab = perda.carregar_tabelas()
    d = base_a[base_a["default_90_12"] == 1].copy()
    d["f_ltv"] = perda.faixa_ltv(d["ltv"])
    d["f_idade"] = perda.faixa_idade(d["idade_veiculo_anos"])
    fator = (d["ead_realizado"] / d["valor_financiado"]).groupby([d["prazo_meses"], d["f_ltv"]]).mean()
    for (prazo, f_ltv), valor in fator.items():
        assert valor == pytest.approx(tab.fator_ead[tab.prazos.index(prazo), f_ltv], abs=0.001)
    lgd_real = d["lgd_realizado"].groupby([d["f_idade"], d["f_ltv"]]).mean()
    for (f_idade, f_ltv), valor in lgd_real.items():
        assert valor == pytest.approx(tab.lgd[f_idade, f_ltv], abs=0.001)
