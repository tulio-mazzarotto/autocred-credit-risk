import json
import re

import pytest

from autocred import dados

SITE = dados.RAIZ_PROJETO / "site"
DOCS = dados.RAIZ_PROJETO / "docs"


@pytest.fixture(scope="module")
def site_dados():
    return json.loads((SITE / "dados.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def textos():
    return json.loads((SITE / "textos.json").read_text(encoding="utf-8"))


def test_dados_tem_todas_as_secoes(site_dados):
    assert {"links", "caso", "modelo", "politica", "simulador", "benchmark", "limites", "oficial"} <= set(site_dados)


def test_simulador_completo(site_dados):
    sim = site_dados["simulador"]
    esperado = len(sim["cortes"]) * len(sim["ajustes_pb"]) * len(sim["entradas"]) * len(sim["fora_perfil"])
    assert esperado == 240 and len(sim["resultados"]) == 240
    for cenarios in sim["resultados"].values():
        assert set(cenarios) == {"brando", "base", "severo"}


def test_politica_vencedora_bate_com_a_politica_final(site_dados):
    sim, final = site_dados["simulador"], site_dados["politica"]["final"]
    vencedora = sim["resultados"][sim["vencedora"]]
    for cenario, metricas in final.items():
        assert vencedora[cenario]["roi"] == pytest.approx(metricas["roi"], abs=1e-4)
        assert vencedora[cenario]["volume_mi"] == pytest.approx(metricas["volume"] / 1e6, abs=0.01)


def test_resultados_oficiais_sao_os_do_arquivo(site_dados):
    oficial = json.loads((DOCS / "resultados_oficiais.json").read_text(encoding="utf-8"))
    assert site_dados["oficial"] == oficial
    assert oficial["auc"] == 0.7711 and oficial["roi"] == 0.1653
    assert "grupos" not in oficial  # a página fala só do nosso grupo


def test_textos_pt_en_com_as_mesmas_chaves(textos):
    assert set(textos["pt"]) == set(textos["en"])
    vazios = [k for idioma in ("pt", "en") for k, v in textos[idioma].items() if not str(v).strip()]
    assert not vazios


def test_toda_chave_usada_na_pagina_existe(textos):
    codigo = (SITE / "index.html").read_text(encoding="utf-8") + (SITE / "app.js").read_text(encoding="utf-8")
    usadas = set(re.findall(r'data-t(?:-html)?="([\w.]+)"', codigo)) | set(re.findall(r'\bt\("([\w.]+)"', codigo))
    faltando = usadas - set(textos["pt"])
    assert usadas and not faltando, faltando


def test_dados_json_valido_para_o_navegador():
    texto = (SITE / "dados.json").read_text(encoding="utf-8")
    assert "NaN" not in texto and "Infinity" not in texto


def test_metas_de_compartilhamento_apontam_para_a_pagina(site_dados):
    # O LinkedIn não roda JavaScript: o endereço do cartão precisa estar escrito no HTML.
    pagina = site_dados["links"]["pagina"]
    html = (SITE / "index.html").read_text(encoding="utf-8")
    assert f'<meta property="og:url" content="{pagina}">' in html
    assert f'<meta property="og:image" content="{pagina}og.png">' in html


def test_scores_estaveis_entre_as_safras(site_dados):
    # A página afirma ausência de drift entre 2023 e 2024: PSI abaixo do limite usual de atenção.
    assert 0 <= site_dados["modelo"]["psi_2023_2024"] < 0.10
