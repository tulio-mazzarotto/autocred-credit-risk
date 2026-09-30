"""Prepara as submissões finais do grupo em ../submissao_final/, com conferências.

Regenera as duas submissões a partir do modelo salvo e da política decidida, confere formato,
conteúdo e coerência com o documento de política (v3) e grava:
  - submissao_modelo_grupo2.csv    (Base B: id_contrato, pd)
  - submissao_politica_grupo2.csv  (Base C: id_proposta, pd, score_1a10, decisao, taxa_am, prazo_meses, pct_entrada_minima)
  - o documento de política v3 (.pdf e .docx)
  - conferencias.txt, com cada checagem e o SHA-256 dos arquivos

Se alguma conferência falhar, nada é gravado.
"""
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lxml import etree

from autocred import dados, escoragem, politica
from autocred.faixas import atribuir_score

RAIZ_DESAFIO = dados.RAIZ_PROJETO.parent
DESTINO = RAIZ_DESAFIO / "submissao_final"
EXEMPLOS = RAIZ_DESAFIO / "entregaveis"
DOCUMENTO = RAIZ_DESAFIO / "documento_politica" / "Politica_Credito_AutoCred_Grupo2_v3"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

conferencias: list[tuple[bool, str]] = []


def conferir(ok, descricao: str) -> None:
    conferencias.append((bool(ok), descricao))


def cabecalho_exemplo(nome: str) -> list[str]:
    return pd.read_csv(EXEMPLOS / nome, nrows=0).columns.tolist()


def regras_decididas() -> pd.DataFrame:
    """A política decidida pelo grupo, como gravada pelo notebook 04 (fonte também da tabela do documento)."""
    registros = json.load(open(dados.PASTA_SAIDAS / "resultados_politica.json", encoding="utf-8"))["regras"]
    colunas = ["aprovar", "taxa_am", "prazo_max", "entrada_min", "aceita_fora_perfil"]
    return pd.DataFrame(registros).set_index("score_1a10")[colunas].sort_index()


def tabela_do_documento() -> dict[int, list[str]]:
    """Lê a tabela da seção 4 do documento v3: score -> textos das células."""
    raiz = etree.fromstring(zipfile.ZipFile(f"{DOCUMENTO}.docx").read("word/document.xml"))
    for tbl in raiz.iter(f"{{{W}}}tbl"):
        linhas = [["".join(tc.itertext()).strip() for tc in tr.findall(f"{{{W}}}tc")] for tr in tbl.findall(f"{{{W}}}tr")]
        if linhas and linhas[0][:2] == ["Score", "Faixa de PD"]:
            return {int(linha[0]): linha for linha in linhas[1:]}
    raise ValueError("tabela de política não encontrada no documento")


def pct_texto(x: float, casas: int) -> str:
    return f"{x * 100:.{casas}f}".replace(".", ",") + "%"


def main() -> None:
    pacote = joblib.load(dados.PASTA_SAIDAS / "modelo_pd.joblib")
    B, C = dados.carregar_base("B"), dados.carregar_base("C")

    # ---------------- Submissão do modelo (Base B) ----------------
    pd_b = escoragem.prever(pacote["modelo_b"], B, pacote["conjunto_b"])
    modelo = pd.DataFrame({"id_contrato": B["id_contrato"].to_numpy(), "pd": np.round(pd_b, 6)})
    conferir(modelo.columns.tolist() == cabecalho_exemplo("submissao_modelo_EXEMPLO.csv"),
             "modelo: colunas idênticas ao exemplo (id_contrato, pd)")
    conferir(len(modelo) == 3000 and modelo["id_contrato"].is_unique, "modelo: 3.000 contratos, sem repetição")
    conferir(np.array_equal(modelo["id_contrato"], B["id_contrato"]), "modelo: ids na mesma ordem da Base B")
    conferir(modelo["pd"].notna().all() and modelo["pd"].between(0, 1, inclusive="neither").all(),
             "modelo: toda PD preenchida e entre 0 e 1")
    anterior = pd.read_csv(dados.PASTA_SAIDAS / "submissao_modelo_base_B.csv")
    conferir(np.allclose(anterior["pd"], modelo["pd"], atol=5e-7),
             "modelo: igual à submissão gerada pelo notebook 03 (mesmo modelo salvo)")

    # ---------------- Submissão da política (Base C) ----------------
    regras = regras_decididas()
    base = politica.preparar_base(pacote["modelo_portatil"], C, pacote["cortes"])
    pol = politica.arquivo_submissao(base, regras)
    conferir(pol.columns.tolist() == cabecalho_exemplo("submissao_politica_EXEMPLO.csv"),
             "política: colunas idênticas ao exemplo, na mesma ordem")
    conferir(len(pol) == 5000 and pol["id_proposta"].is_unique, "política: 5.000 propostas, sem repetição")
    conferir(np.array_equal(pol["id_proposta"], C["id_proposta"]), "política: ids na mesma ordem da Base C")
    conferir(set(pol["decisao"]) <= {"APROVAR", "NEGAR"}, "política: decisão só APROVAR ou NEGAR")
    negar, aprovar = pol["decisao"] == "NEGAR", pol["decisao"] == "APROVAR"
    campos = ["taxa_am", "prazo_meses", "pct_entrada_minima"]
    conferir(pol.loc[negar, campos].isna().all().all() and pol.loc[aprovar, campos].notna().all().all(),
             "política: NEGAR sem taxa, prazo e entrada; APROVAR com os três preenchidos")
    conferir(pol.loc[aprovar, "taxa_am"].max() <= politica.TETO_TAXA_AM,
             f"política: taxa máxima {pct_texto(pol.loc[aprovar, 'taxa_am'].max(), 1)} a.m., dentro do teto de 3,5%")
    conferir(pol.loc[aprovar, "prazo_meses"].isin([24, 36, 48, 60]).all(), "política: prazos só 24, 36, 48 ou 60 meses")
    pedido = C.set_index("id_proposta").loc[pol.loc[aprovar, "id_proposta"], "prazo_desejado_meses"].to_numpy()
    conferir(np.array_equal(pol.loc[aprovar, "prazo_meses"].to_numpy(dtype=int), pedido),
             "política: todo aprovado recebe o prazo que pediu (60 meses é teto, não oferta)")
    conferir(pol["pct_entrada_minima"].dropna().between(0, 1).all(), "política: entrada mínima entre 0% e 100%")
    try:
        politica.validar_coerencia(pol, base, regras, pacote["cortes"])
        conferir(True, "política: cada linha segue a regra da sua faixa; score coerente com a PD e os cortes")
    except ValueError as erro:
        conferir(False, f"política: coerência com a tabela de regras ({erro})")
    anterior = pd.read_csv(dados.PASTA_SAIDAS / "politica_base_C.csv")
    iguais = all(
        ((anterior[c].isna() & pol[c].isna()) | (anterior[c] == pol[c])).all() for c in pol.columns if c != "pd"
    ) and np.allclose(anterior["pd"], pol["pd"], atol=5e-7)
    conferir(iguais, "política: igual à gerada pelo notebook 04")
    conferir(round(aprovar.mean(), 3) == 0.448,
             f"política: aprovação de {pct_texto(aprovar.mean(), 1)}, a mesma do documento (44,8%)")

    # ---------------- Coerência com a tabela do documento v3 ----------------
    documento = tabela_do_documento()
    divergencias = []
    for score, grupo in pol.groupby("score_1a10"):
        celulas = documento[int(score)]
        decisao = "Aprovar" if (grupo["decisao"] == "APROVAR").all() else "Negar"
        if celulas[2] != decisao:
            divergencias.append(f"score {score}: decisão {celulas[2]} x {decisao}")
        if decisao == "Aprovar":
            ap = grupo[grupo["decisao"] == "APROVAR"]
            if {pct_texto(t, 1) for t in ap["taxa_am"]} != {celulas[3]}:
                divergencias.append(f"score {score}: taxa {celulas[3]} x {sorted(set(ap['taxa_am']))}")
            if {pct_texto(e, 0) for e in ap["pct_entrada_minima"]} != {celulas[5]}:
                divergencias.append(f"score {score}: entrada {celulas[5]} x {sorted(set(ap['pct_entrada_minima']))}")
        limites = [float(v.replace(",", ".")) / 100 for v in re.findall(r"\d+,\d+", celulas[1])]
        if "abaixo" in celulas[1]:
            fora = grupo["pd"] >= limites[0] + 5e-5
        elif "ou mais" in celulas[1]:
            fora = grupo["pd"] < limites[0] - 5e-5
        else:
            fora = (grupo["pd"] < limites[0] - 5e-5) | (grupo["pd"] >= limites[1] + 5e-5)
        if fora.any():
            divergencias.append(f"score {score}: {int(fora.sum())} PDs fora da faixa '{celulas[1]}'")
    conferir(not divergencias, "documento v3: decisão, taxa, entrada e faixa de PD da tabela batem com a submissão"
             + ("" if not divergencias else " -> " + "; ".join(divergencias)))
    conferir(len(documento) == 10 and np.array_equal(atribuir_score(pol["pd"], pacote["cortes"]), pol["score_1a10"]),
             "documento v3: dez faixas, e o score de cada proposta sai dos mesmos cortes")

    # ---------------- Resultado ----------------
    relatorio = [f"{'OK  ' if ok else 'FALHA'}  {texto}" for ok, texto in conferencias]
    print("\n".join(relatorio))
    if not all(ok for ok, _ in conferencias):
        sys.exit("\nAlguma conferência falhou: nada foi gravado.")

    DESTINO.mkdir(exist_ok=True)
    modelo.to_csv(DESTINO / "submissao_modelo_grupo2.csv", index=False, float_format="%.6f", lineterminator="\n")
    pol.to_csv(DESTINO / "submissao_politica_grupo2.csv", index=False, lineterminator="\n")
    for extensao in (".pdf", ".docx"):
        shutil.copy2(f"{DOCUMENTO}{extensao}", DESTINO / f"{Path(DOCUMENTO).name}{extensao}")

    linhas_hash = []
    for arquivo in sorted(p for p in DESTINO.iterdir() if p.name != "conferencias.txt"):
        linhas_hash.append(f"{hashlib.sha256(arquivo.read_bytes()).hexdigest()}  {arquivo.name}")
    (DESTINO / "conferencias.txt").write_text(
        "Conferências das submissões do Grupo 2\n\n" + "\n".join(relatorio)
        + "\n\nSHA-256 dos arquivos entregues\n" + "\n".join(linhas_hash) + "\n", encoding="utf-8")
    print("\nGravado em", DESTINO)
    print("\n".join(linhas_hash))


if __name__ == "__main__":
    main()
