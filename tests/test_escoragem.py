import numpy as np
import pandas as pd
import pytest

from autocred import escoragem, features

CORTES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) / 2


class _PDMetadeDoLTV:
    """Modelo de mentira: PD = LTV / 2. Determinístico e sensível à entrada exigida."""

    def predict_proba(self, X):
        p = X["ltv"].to_numpy(dtype=float) / 2
        return np.column_stack([1 - p, p])


def test_escorar_base_b(base_b):
    r = escoragem.escorar(_PDMetadeDoLTV(), base_b, CORTES)
    assert list(r.columns) == ["pd", "score_1a10", "fator_ead", "ead", "lgd", "perda_esperada", "perda_esperada_pct"]
    assert r.index.equals(base_b.index)
    np.testing.assert_allclose(r["pd"], base_b["ltv"] / 2)
    np.testing.assert_allclose(r["perda_esperada"], r["pd"] * r["ead"] * r["lgd"])
    assert r["score_1a10"].between(1, 10).all()


def test_prever_pd_sem_mudar_oferta_reproduz_escoragem_padrao(base_c):
    r = escoragem.prever_pd(_PDMetadeDoLTV(), base_c, CORTES)
    padrao = escoragem.escorar(_PDMetadeDoLTV(), features.oferta_no_formato_a(base_c), CORTES)
    assert r.columns[0] == "id_proposta"
    np.testing.assert_allclose(r["pd"], padrao["pd"])
    np.testing.assert_allclose(r["perda_esperada"], padrao["perda_esperada"])
    assert r["fora_perfil_historico"].mean() == pytest.approx(0.395, abs=0.001)


def test_entrada_minima_reduz_pd_de_quem_foi_afetado(base_c):
    antes = escoragem.prever_pd(_PDMetadeDoLTV(), base_c, CORTES)
    depois = escoragem.prever_pd(_PDMetadeDoLTV(), base_c, CORTES, pct_entrada_minima=0.30)
    afetados = (base_c["pct_entrada_desejada"] < 0.30).to_numpy()
    assert np.all(depois["pd"].to_numpy()[afetados] < antes["pd"].to_numpy()[afetados])
    np.testing.assert_allclose(depois["pd"].to_numpy()[~afetados], antes["pd"].to_numpy()[~afetados])


def test_fora_perfil_nao_marca_bureau_ausente():
    X = pd.DataFrame({
        "score_bureau": [np.nan, 450.0, 700.0, 700.0, 700.0],
        "qtd_restricoes_ativas": [0, 0, 3, 0, 0],
        "ltv": [0.8, 0.8, 0.8, 0.96, 0.95],
    })
    assert escoragem.fora_perfil_historico(X).tolist() == [False, True, True, True, False]


def test_submissao_b_ok(base_b, tmp_path):
    p = np.linspace(0.01, 0.5, len(base_b))
    sub = escoragem.salvar_submissao_b(base_b["id_contrato"], p, base_b, tmp_path / "sub.csv")
    assert len(sub) == 3000
    segunda_linha = (tmp_path / "sub.csv").read_text(encoding="utf-8").splitlines()[1]
    assert segunda_linha == "T000001,0.010000"


def test_submissao_b_ordem_errada(base_b):
    sub = pd.DataFrame({"id_contrato": base_b["id_contrato"][::-1].to_numpy(), "pd": 0.1})
    with pytest.raises(ValueError, match="ordem"):
        escoragem.validar_submissao_b(sub, base_b)


def test_submissao_b_pd_invalida(base_b):
    sub = pd.DataFrame({"id_contrato": base_b["id_contrato"].to_numpy(), "pd": 0.0})
    with pytest.raises(ValueError, match="entre 0 e 1"):
        escoragem.validar_submissao_b(sub, base_b)
