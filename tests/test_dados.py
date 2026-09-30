import os

import pytest

from autocred import dados


def test_bases_tem_tamanho_esperado(base_a, base_b, base_c):
    assert base_a.shape[0] == 10_000
    assert base_b.shape[0] == 3_000
    assert base_c.shape[0] == 5_000


def test_ano_da_base_a(base_a):
    anos = dados.ano(base_a)
    assert sorted(anos.unique()) == [2022, 2023, 2024]


def test_colunas_entrada_existem_nas_bases_a_e_b(base_a, base_b):
    for conjunto in ("portatil", "completo"):
        colunas = dados.colunas_entrada(conjunto)
        assert set(colunas) <= set(base_a.columns)
        assert set(colunas) <= set(base_b.columns)


def test_portatil_nao_tem_taxa_e_completo_tem():
    assert "taxa_juros_am" not in dados.colunas_entrada("portatil")
    assert "taxa_juros_am" in dados.colunas_entrada("completo")
    assert "taxa_juros_am" in dados.numericas_modelo("completo")


def test_colunas_entrada_sem_proibidas():
    for conjunto in ("portatil", "completo"):
        dados.validar_sem_proibidas(dados.colunas_entrada(conjunto))


def test_trava_de_proibidas_dispara():
    with pytest.raises(ValueError, match="qtd_parcelas_em_atraso_12m"):
        dados.validar_sem_proibidas(["score_bureau", "qtd_parcelas_em_atraso_12m"])


def test_numericas_modelo_inclui_indicadores():
    numericas = dados.numericas_modelo("portatil")
    assert numericas[-3:] == dados.INDICADORES
    assert dados.INDICADORES == [
        "ausente_score_bureau",
        "ausente_renda_mensal_declarada",
        "ausente_tempo_emprego_meses",
    ]


def test_conjunto_invalido():
    with pytest.raises(ValueError):
        dados.colunas_entrada("tudo")


def test_pasta_das_bases_padrao_prefere_a_do_proprio_projeto():
    """No repositório público as bases vêm junto, em bases/; aqui, ficam na pasta do desafio."""
    import subprocess
    import sys

    ambiente = {k: v for k, v in os.environ.items() if k != "AUTOCRED_BASES"} | {"PYTHONIOENCODING": "utf-8"}
    codigo = "from autocred import dados; print(dados.PASTA_BASES)"
    saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, encoding="utf-8", env=ambiente,
                           check=True)
    no_projeto = dados.RAIZ_PROJETO / "bases"
    esperado = no_projeto if no_projeto.is_dir() else dados.RAIZ_PROJETO.parent / "bases"
    assert saida.stdout.strip() == str(esperado)


def test_pasta_das_bases_pode_vir_do_ambiente(tmp_path):
    """O kit do colega roda fora deste projeto: as pastas vêm por variável de ambiente."""
    import subprocess
    import sys

    codigo = "from autocred import dados; print(dados.PASTA_BASES); print(dados.PASTA_SAIDAS)"
    ambiente = {**os.environ, "AUTOCRED_BASES": str(tmp_path / "b"), "AUTOCRED_SAIDAS": str(tmp_path / "s"),
                "PYTHONIOENCODING": "utf-8"}
    saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, encoding="utf-8", env=ambiente,
                           check=True)
    assert saida.stdout.splitlines() == [str(tmp_path / "b"), str(tmp_path / "s")]
