import numpy as np
import pandas as pd
import pytest

from autocred import avaliacao

Y = [0, 0, 1, 1]
P = [0.10, 0.40, 0.35, 0.80]


def test_auc_e_ks_exemplo_manual():
    # pares (mau, bom): 0.35>0.10 sim, 0.35>0.40 não, 0.80>0.10 sim, 0.80>0.40 sim -> 3/4
    assert avaliacao.auc(Y, P) == pytest.approx(0.75)
    # corte em 0.80: 50% dos maus e 0% dos bons acima -> KS = 0.5
    assert avaliacao.ks(Y, P) == pytest.approx(0.5)


def test_ks_separacao_perfeita():
    assert avaliacao.ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_bootstrap_auc_contem_ponto_e_e_reprodutivel():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = rng.binomial(1, p)
    ic = avaliacao.bootstrap_auc(y, p, n=200)
    assert ic[0] < avaliacao.auc(y, p) < ic[1]
    assert ic == avaliacao.bootstrap_auc(y, p, n=200)


def test_diferenca_de_modelos_iguais_e_zero():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=500)
    y = rng.binomial(1, p)
    assert avaliacao.bootstrap_diferenca_auc(y, p, p, n=50) == (0.0, 0.0, 0.0)


def test_mascaras_janela():
    anos = pd.Series([2022, 2023, 2024])
    tr, te = avaliacao.mascaras_janela(anos, "J1")
    assert tr.tolist() == [True, False, False] and te.tolist() == [False, True, False]
    tr, te = avaliacao.mascaras_janela(anos, "J2")
    assert tr.tolist() == [True, True, False] and te.tolist() == [False, False, True]


def test_avaliar_temporal_treina_so_com_anos_de_treino():
    tamanhos = []

    class Espiao:
        def fit(self, X, y):
            tamanhos.append(len(X))
            return self

        def predict_proba(self, X):
            p = X["x"].to_numpy(dtype=float)
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"x": P * 3})
    y = np.array(Y * 3)
    anos = pd.Series([2022] * 4 + [2023] * 4 + [2024] * 4)
    metricas, previsoes = avaliacao.avaliar_temporal(Espiao, X, y, anos)
    assert tamanhos == [4, 8]
    assert metricas["auc_J1"] == pytest.approx(0.75)
    assert metricas["auc_medio"] == pytest.approx(0.75)
    assert len(previsoes["J2"]) == 4


def _candidatos(linhas):
    return pd.DataFrame(linhas, columns=["candidato", "complexidade", "auc_medio"])


def test_campeao_troca_so_com_ganho_minimo():
    c = _candidatos([("logistica", 0, 0.700), ("mono", 1, 0.703), ("xgboost", 2, 0.707), ("ensemble", 3, 0.709)])
    assert avaliacao.escolher_campeao(c)["candidato"] == "xgboost"


def test_campeao_empate_fica_com_o_mais_simples():
    c = _candidatos([("logistica", 0, 0.700), ("mono", 1, 0.702), ("lightgbm", 2, 0.703)])
    assert avaliacao.escolher_campeao(c)["candidato"] == "logistica"


def test_campeao_no_mesmo_nivel_pega_o_melhor():
    c = _candidatos([("logistica", 0, 0.700), ("xgboost", 2, 0.707), ("lightgbm", 2, 0.712)])
    assert avaliacao.escolher_campeao(c)["candidato"] == "lightgbm"


def _dados_calibracao(compressao):
    rng = np.random.default_rng(2)
    p_real = rng.uniform(0.01, 0.30, size=20_000)
    y = rng.binomial(1, p_real)
    logit = np.log(p_real / (1 - p_real))
    p_modelo = 1 / (1 + np.exp(-(compressao * logit)))
    return y, p_modelo


def test_inclinacao_perto_de_1_quando_calibrado():
    y, p = _dados_calibracao(1.0)
    assert 0.85 < avaliacao.inclinacao_calibracao(y, p) < 1.15


def test_platt_corrige_previsao_comprimida():
    y, p = _dados_calibracao(0.5)  # PDs "achatadas": inclinação real perto de 2
    a, b = avaliacao.ajustar_platt(y, p)
    assert a > 1.5
    corrigida = avaliacao.aplicar_platt(p, a, b)
    assert 0.85 < avaliacao.inclinacao_calibracao(y, corrigida) < 1.15
    assert avaliacao.auc(y, corrigida) == pytest.approx(avaliacao.auc(y, p))


class _Identidade:
    def predict_proba(self, X):
        p = X["x"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_dependencia_parcial_numerica():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 101), "cat": ["a"] * 101})
    r = avaliacao.dependencia_parcial(_Identidade(), X, "x", n_pontos=20)
    assert len(r) == 20
    np.testing.assert_allclose(r["pd_media"], r["valor"])


def test_dependencia_parcial_categorica():
    X = pd.DataFrame({"x": [0.2, 0.4], "cat": ["a", "b"]})
    r = avaliacao.dependencia_parcial(_Identidade(), X, "cat")
    assert r["valor"].tolist() == ["a", "b"]
    np.testing.assert_allclose(r["pd_media"], 0.3)


def test_psi_zero_quando_as_distribuicoes_sao_iguais():
    assert avaliacao.psi([1, 2, 3, 4], [4, 3, 2, 1], n_faixas=4) == pytest.approx(0.0)


def test_psi_exemplo_manual():
    # referência 50%/50% e atual 25%/75%: (0,25-0,5)·ln(0,5) + (0,75-0,5)·ln(1,5) = 0,1733 + 0,1014
    esperado = -0.25 * np.log(0.5) + 0.25 * np.log(1.5)
    assert avaliacao.psi([1, 1, 2, 2], [1, 2, 2, 2], n_faixas=2) == pytest.approx(esperado)
    assert avaliacao.psi([1, 2, 2, 2], [1, 1, 2, 2], n_faixas=2) == pytest.approx(esperado)  # simétrico
