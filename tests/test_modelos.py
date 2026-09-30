import numpy as np
import pytest
from sklearn.ensemble import VotingClassifier

from autocred import dados, modelos
from autocred.avaliacao import auc


@pytest.fixture(scope="module")
def particao(base_a):
    X = base_a[dados.colunas_entrada("completo")]
    y = base_a[dados.ALVO].to_numpy()
    return X.iloc[:3000], y[:3000], X.iloc[3000:3500], y[3000:3500]


def _portatil(X):
    return X[dados.colunas_entrada("portatil")]


@pytest.mark.parametrize("tipo", modelos.TIPOS)
def test_cada_tipo_treina_e_preve_probabilidades(particao, tipo):
    X_tr, y_tr, X_te, _ = particao
    params = modelos.amostrar_configs(tipo)[0]
    p = modelos.construir_pipeline(tipo, params).fit(_portatil(X_tr), y_tr).predict_proba(_portatil(X_te))[:, 1]
    assert p.shape == (500,)
    assert np.all((p > 0) & (p < 1))


def test_amostrar_configs_reprodutivel():
    assert len(modelos.amostrar_configs("xgboost")) == 25
    assert modelos.amostrar_configs("lightgbm") == modelos.amostrar_configs("lightgbm")
    cs = [c["C"] for c in modelos.amostrar_configs("logistica")]
    assert len(cs) == 25 and cs == sorted(cs)


def test_conjunto_completo_usa_taxa(particao):
    X_tr, y_tr, _, _ = particao
    params = modelos.amostrar_configs("xgboost")[0]
    portatil = modelos.construir_pipeline("xgboost", params, "portatil").fit(_portatil(X_tr), y_tr)
    completo = modelos.construir_pipeline("xgboost", params, "completo").fit(X_tr, y_tr)
    assert portatil.named_steps["modelo"].n_features_in_ == 21
    assert completo.named_steps["modelo"].n_features_in_ == 22


def test_restricoes_alinhadas_com_as_colunas():
    r = modelos.restricoes_monotonicas("portatil")
    nomes = dados.numericas_modelo("portatil") + dados.CATEGORICAS
    assert len(r) == len(nomes) == 21
    assert r[nomes.index("score_bureau")] == -1
    assert r[nomes.index("ltv")] == 1
    assert r[nomes.index("idade_cliente")] == 0


@pytest.mark.parametrize("tipo", ["xgboost", "lightgbm"])
def test_modelo_monotonico_respeita_direcoes(particao, tipo):
    X_tr, y_tr, X_te, _ = particao
    params = modelos.amostrar_configs(tipo)[0]
    pipe = modelos.construir_pipeline(tipo, params, monotonico=True).fit(_portatil(X_tr), y_tr)
    base = _portatil(X_te).iloc[:200]
    grades = {"qtd_restricoes_ativas": [0, 1, 2], "score_bureau": [460.0, 600.0, 800.0, 1000.0], "ltv": [0.3, 0.6, 0.9]}
    for coluna, grade in grades.items():
        previsoes = []
        for valor in grade:
            X = base.copy()
            X[coluna] = valor
            previsoes.append(pipe.predict_proba(X)[:, 1])
        difs = np.diff(np.vstack(previsoes), axis=0) * modelos.MONOTONIA[coluna]
        assert np.all(difs >= -1e-12), coluna


def test_ensemble_e_a_media_dos_componentes(particao):
    X_tr, y_tr, X_te, _ = particao
    componentes = [
        {"tipo": "logistica", "params": {"C": 0.1}, "monotonico": False},
        {"tipo": "lightgbm", "params": modelos.amostrar_configs("lightgbm")[0], "monotonico": False},
    ]
    ens = modelos.construir_modelo({"tipo": "ensemble", "componentes": componentes})
    assert isinstance(ens, VotingClassifier)
    p_ens = ens.fit(_portatil(X_tr), y_tr).predict_proba(_portatil(X_te))[:, 1]
    p_sep = [modelos.construir_modelo(c).fit(_portatil(X_tr), y_tr).predict_proba(_portatil(X_te))[:, 1] for c in componentes]
    np.testing.assert_allclose(p_ens, np.mean(p_sep, axis=0))


def test_modelo_calibrado_preserva_ordem(particao):
    X_tr, y_tr, X_te, y_te = particao
    pipe = modelos.construir_pipeline("logistica", {"C": 0.1}).fit(_portatil(X_tr), y_tr)
    calibrado = modelos.ModeloCalibrado(pipe, 1.5, 0.2)
    p0 = pipe.predict_proba(_portatil(X_te))[:, 1]
    p1 = calibrado.predict_proba(_portatil(X_te))[:, 1]
    assert not np.allclose(p0, p1)
    assert auc(y_te, p1) == pytest.approx(auc(y_te, p0))


def test_treinar_final_logistica(base_a):
    anos = dados.ano(base_a)
    descricao = {"tipo": "logistica", "params": {"C": 0.1}, "monotonico": False}
    modelo, info, previsoes = modelos.treinar_final(descricao, base_a, base_a[dados.ALVO], anos)
    assert {"auc_J1", "auc_J2", "auc_medio", "inclinacao_J2", "calibrado", "conjunto"} <= set(info)
    assert isinstance(info["calibrado"], bool)
    assert len(previsoes["J2"]) == int((anos == 2024).sum())
    p = modelo.predict_proba(base_a[dados.colunas_entrada("portatil")])[:, 1]
    assert np.all((p > 0) & (p < 1))


def test_complexidade():
    assert modelos.complexidade({"tipo": "logistica"}) == 0
    assert modelos.complexidade({"tipo": "xgboost", "monotonico": True}) == 1
    assert modelos.complexidade({"tipo": "random_forest", "monotonico": False}) == 2
    assert modelos.complexidade({"tipo": "ensemble", "componentes": []}) == 3
