"""Leitura das bases do desafio e listas de colunas do modelo."""
import os
from pathlib import Path

import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
# Fora deste projeto (o kit do colega de política, o Databricks), a pasta das bases
# e a de saídas mudam de lugar: as variáveis de ambiente abaixo permitem apontá-las.
# No repositório público as bases vêm junto, em bases/; neste projeto, ficam na pasta do desafio.
_BASES_NO_PROJETO = RAIZ_PROJETO / "bases"
PASTA_BASES = Path(
    os.environ.get("AUTOCRED_BASES")
    or (_BASES_NO_PROJETO if _BASES_NO_PROJETO.is_dir() else RAIZ_PROJETO.parent / "bases")
)
PASTA_SAIDAS = Path(os.environ.get("AUTOCRED_SAIDAS") or RAIZ_PROJETO / "saidas")

ARQUIVOS = {
    "A": "base_A_autocred_base_desenvolvimento.csv",
    "B": "base_B_autocred_base_teste_modelo.csv",
    "C": "base_C_autocred_base_politica.csv",
}
ARQ_PARAMETROS = "AutoCred_parametros_ead_lgd.xlsx"

ALVO = "default_90_12"
COL_DATA = "data_originacao"

# Nunca entram no modelo: vazamento, alvo, realizados, identificadores e datas.
PROIBIDAS = frozenset({
    "qtd_parcelas_em_atraso_12m",  # apurada DEPOIS da concessão (vazamento)
    "default_90_12",
    "mes_default",
    "ead_realizado",
    "lgd_realizado",
    "perda_financeira",
    "id_contrato",
    "id_proposta",
    "data_originacao",
    "data_proposta",
    "ano_modelo",  # redundante com a idade do veículo e deriva com o calendário
})

NUMERICAS = [
    "valor_bem",
    "valor_entrada",
    "valor_financiado",
    "ltv",
    "prazo_meses",
    "parcela_mensal",
    "comprometimento_renda",
    "idade_veiculo_anos",
    "idade_cliente",
    "renda_mensal_declarada",
    "tempo_emprego_meses",
    "score_bureau",
    "qtd_restricoes_ativas",
    "qtd_consultas_bureau_3m",
]
CATEGORICAS = ["canal_originacao", "ocupacao", "tipo_residencia", "possui_avalista"]
COM_AUSENTES = ["score_bureau", "renda_mensal_declarada", "tempo_emprego_meses"]
INDICADORES = [f"ausente_{c}" for c in COM_AUSENTES]
CONJUNTOS = ("portatil", "completo")


def _checar_conjunto(conjunto: str) -> None:
    if conjunto not in CONJUNTOS:
        raise ValueError(f"conjunto deve ser um de {CONJUNTOS}, recebi {conjunto!r}")


def colunas_entrada(conjunto: str) -> list[str]:
    """Colunas brutas que o Pipeline recebe (os indicadores são criados dentro dele)."""
    _checar_conjunto(conjunto)
    extra = ["taxa_juros_am"] if conjunto == "completo" else []
    return NUMERICAS + extra + CATEGORICAS


def numericas_modelo(conjunto: str) -> list[str]:
    """Numéricas na ordem usada pelo Pipeline: brutas, taxa (se completo) e indicadores."""
    _checar_conjunto(conjunto)
    extra = ["taxa_juros_am"] if conjunto == "completo" else []
    return NUMERICAS + extra + INDICADORES


def validar_sem_proibidas(colunas) -> None:
    """Trava contra vazamento: falha se alguma coluna proibida chegar ao modelo."""
    proibidas = sorted(set(colunas) & PROIBIDAS)
    if proibidas:
        raise ValueError(f"Colunas proibidas no modelo: {proibidas}")


def carregar_base(nome: str) -> pd.DataFrame:
    """Lê a Base A, B ou C da pasta do desafio."""
    return pd.read_csv(PASTA_BASES / ARQUIVOS[nome], encoding="utf-8")


def ano(df: pd.DataFrame) -> pd.Series:
    """Ano de originação (safra) como inteiro."""
    return df[COL_DATA].str[:4].astype(int)
