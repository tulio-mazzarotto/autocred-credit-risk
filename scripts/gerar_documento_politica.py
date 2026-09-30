"""Gera o documento de política (v2 e v3) a partir do template oficial do desafio.

O texto é o do documento do Fábio (politica_credito_autocred_preenchida.pdf) com as melhorias
revisadas em 23/09 (v2) e a seção de benchmarking de taxas com dados do Banco Central (v3).
Para cada versão saem duas cópias, em .docx e .pdf:
  - limpa: pronta para entregar;
  - com as alterações daquela versão grifadas em amarelo: para o grupo revisar o que mudou.

A tabela de faixas vem dos arquivos de resultado; os números do texto são conferidos contra eles.
Os números do benchmarking vêm de docs/benchmark/benchmark_bcb.json (scripts/buscar_benchmark_bcb.py).
"""
import json
import os
import shutil
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from lxml import etree

from autocred import dados

RAIZ_DESAFIO = dados.RAIZ_PROJETO.parent
TEMPLATE = RAIZ_DESAFIO / "entregaveis" / "template_documento_politica.docx"
DESTINO = RAIZ_DESAFIO / "documento_politica"
BENCHMARK = dados.RAIZ_PROJETO / "docs" / "benchmark"
VERSOES = {2: "Politica_Credito_AutoCred_Grupo2_v2", 3: "Politica_Credito_AutoCred_Grupo2_v3"}
# LibreOffice converte o .docx em PDF. Em outra máquina, aponte AUTOCRED_SOFFICE ou deixe o soffice no PATH.
SOFFICE = Path(os.environ.get("AUTOCRED_SOFFICE") or shutil.which("soffice")
               or "C:/Program Files/LibreOffice/program/soffice.exe")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
TAM_TABELA = 19  # 9,5 pt, o tamanho que o template usa nas tabelas


def w(tag):
    return f"{{{W}}}{tag}"


def S(texto, novo=False, negrito=False):
    """Um trecho de texto. `novo` = versão em que o trecho entrou (True = v2; 3 = v3; False = texto do Fábio)."""
    versao = 2 if novo is True else (novo or None)
    return (texto, versao, negrito)


def num(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x, casas=1):
    return f"{x * 100:.{casas}f}".replace(".", ",") + "%"


# --------------------------------------------------------------------------- #
# Manipulação do XML
# --------------------------------------------------------------------------- #
def novo_run(texto, negrito=False, destacar=False, tamanho=None, cor=None):
    r = etree.Element(w("r"))
    rpr = etree.SubElement(r, w("rPr"))
    if negrito:
        etree.SubElement(rpr, w("b"))
        etree.SubElement(rpr, w("bCs"))
    if cor:
        etree.SubElement(rpr, w("color")).set(w("val"), cor)
    if tamanho:
        etree.SubElement(rpr, w("sz")).set(w("val"), str(tamanho))
        etree.SubElement(rpr, w("szCs")).set(w("val"), str(tamanho))
    if destacar:
        etree.SubElement(rpr, w("highlight")).set(w("val"), "yellow")
    if len(rpr) == 0:
        r.remove(rpr)
    t = etree.SubElement(r, w("t"))
    t.text = texto
    t.set(XML_SPACE, "preserve")
    return r


def alinhar(p, valor):
    ppr = p.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        p.insert(0, ppr)
    for jc in ppr.findall(w("jc")):
        ppr.remove(jc)
    etree.SubElement(ppr, w("jc")).set(w("val"), valor)


def preencher(p, segmentos, destacar, tamanho=None, cor=None, alinhamento=None):
    """Troca o conteúdo do parágrafo pelos segmentos, preservando as propriedades de parágrafo."""
    for filho in list(p):
        if filho.tag != w("pPr"):
            p.remove(filho)
    if alinhamento:
        alinhar(p, alinhamento)
    for texto, versao, negrito in segmentos:
        grifar = destacar is not None and versao == destacar
        p.append(novo_run(texto, negrito=negrito, destacar=grifar, tamanho=tamanho, cor=cor))


def paragrafo_como(modelo, segmentos, destacar, **kw):
    novo = etree.Element(w("p"))
    ppr = modelo.find(w("pPr"))
    if ppr is not None:
        novo.append(deepcopy(ppr))
    preencher(novo, segmentos, destacar, **kw)
    return novo


def inserir_depois(referencia, paragrafos):
    for p in reversed(paragrafos):
        referencia.addnext(p)


def trocar_texto(p, texto):
    """Troca o texto mantendo a formatação do primeiro trecho (cabeçalhos do template)."""
    ts = p.findall(f".//{w('t')}")
    ts[0].text = texto
    for t in ts[1:]:
        t.text = ""


def grifar_existente(p):
    """Grifa um parágrafo que já existe no template (a seção 8 não estava no documento do Fábio)."""
    for r in p.findall(w("r")):
        rpr = r.find(w("rPr"))
        if rpr is None:
            rpr = etree.Element(w("rPr"))
            r.insert(0, rpr)
        etree.SubElement(rpr, w("highlight")).set(w("val"), "yellow")


def apagar(p):
    p.getparent().remove(p)


def copia_sem_ids(elemento):
    """Cópia profunda sem os identificadores w14 (paraId/textId), que precisam ser únicos no documento."""
    copia = deepcopy(elemento)
    for el in copia.iter():
        for atributo in (f"{{{W14}}}paraId", f"{{{W14}}}textId"):
            el.attrib.pop(atributo, None)
    return copia


def definir_larguras(tbl, proporcoes):
    grade = tbl.find(w("tblGrid")).findall(w("gridCol"))
    total = sum(int(c.get(w("w"))) for c in grade)
    larguras = [round(total * p) for p in proporcoes]
    for c in grade:
        tbl.find(w("tblGrid")).remove(c)
    for largura in larguras:
        etree.SubElement(tbl.find(w("tblGrid")), w("gridCol")).set(w("w"), str(largura))
    for tr in tbl.findall(w("tr")):
        for tc, largura in zip(tr.findall(w("tc")), larguras):
            tcpr = tc.find(w("tcPr"))
            if tcpr is None:
                tcpr = etree.Element(w("tcPr"))
                tc.insert(0, tcpr)
            tcw = tcpr.find(w("tcW"))
            if tcw is None:
                tcw = etree.Element(w("tcW"))
                tcpr.insert(0, tcw)
            tcw.set(w("w"), str(largura))
            tcw.set(w("type"), "dxa")


MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def mes_ano(aaaa_mm: str) -> str:
    ano, mes = aaaa_mm.split("-")
    return f"{MESES[int(mes) - 1]}/{ano}"


def tabela_de(p):
    return next(a for a in p.iterancestors() if a.tag == w("tbl"))


# --------------------------------------------------------------------------- #
# Dados que alimentam a tabela de faixas e a conferência dos números
# --------------------------------------------------------------------------- #
def carregar_numeros():
    saidas = dados.PASTA_SAIDAS
    modelo = json.load(open(saidas / "resultados_modelo.json", encoding="utf-8"))
    politica = json.load(open(saidas / "resultados_politica.json", encoding="utf-8"))
    faixas = pd.read_csv(saidas / "faixas_score.csv", index_col=0)
    cortes = joblib.load(saidas / "modelo_pd.joblib")["cortes"]
    bench = json.load(open(BENCHMARK / "benchmark_bcb.json", encoding="utf-8"))
    bench["instituicoes_tipicas"] = pd.read_csv(BENCHMARK / "benchmark_instituicoes.csv")["taxa_tipica"].tolist()
    bench["taxa_antiga_autocred"] = float(dados.carregar_base("A")["taxa_juros_am"].mean()) * 100
    bench["inadimplencia_autocred"] = float(dados.carregar_base("A")[dados.ALVO].mean()) * 100
    return modelo, politica, faixas, cortes, bench


def conferir(modelo, politica, faixas):
    """Os números digitados no texto precisam bater com os arquivos de resultado."""
    camp = modelo["campeao"]
    cen = {c["cenario"]: c for c in politica["final"]}
    esperado = {
        "AUC 2023": (f"{camp['auc_J1']:.3f}", "0.724"),
        "AUC 2024": (f"{camp['auc_J2']:.3f}", "0.750"),
        "AUC médio": (f"{camp['auc_medio']:.3f}", "0.737"),
        "KS 2024": (f"{camp['ks_J2']:.3f}", "0.404"),
        "aprovação": (pct(cen["base"]["aprovacao"]), "44,8%"),
        "volume base": (f"{cen['base']['volume'] / 1e6:.1f}", "52.6"),
        "volume severo": (f"{cen['severo']['volume'] / 1e6:.1f}", "41.2"),
        "inad. base": (pct(cen["base"]["inadimplencia"]), "6,5%"),
        "inad. severo": (pct(cen["severo"]["inadimplencia"]), "7,5%"),
        "ROI base": (pct(cen["base"]["roi"]), "16,4%"),
        "ROI severo": (pct(cen["severo"]["roi"]), "16,0%"),
        "PD média C": (pct(modelo["pd_media_C"]), "14,4%"),
        "PD média B": (pct(modelo["pd_media_B"]), "7,8%"),
        "fora do perfil": (pct(modelo["pct_fora_perfil_C"]), "39,5%"),
        "observada score 1": (pct(faixas.loc[1, "inadimplencia_observada_J2"]), "24,0%"),
        "observada score 10": (pct(faixas.loc[10, "inadimplencia_observada_J2"]), "2,4%"),
    }
    erros = [f"{nome}: arquivo {real} x texto {texto}" for nome, (real, texto) in esperado.items() if real != texto]
    if erros:
        raise ValueError("Números do texto divergem dos resultados:\n" + "\n".join(erros))


# --------------------------------------------------------------------------- #
# O documento
# --------------------------------------------------------------------------- #
def montar(versao: int, destacar, modelo, politica, faixas, cortes, bench) -> bytes:
    """`destacar` = versão cujas alterações saem grifadas (None = cópia limpa)."""
    with zipfile.ZipFile(TEMPLATE) as z:
        raiz = etree.fromstring(z.read("word/document.xml"))
    P = list(raiz.iter(w("p")))

    # Cabeçalho: grupo preenchido, instruções fora.
    trocar_texto(P[1], "Modelo de documento · Grupo 2 · Desafio de Risco de Crédito · setembro de 2026")
    apagar(P[2])

    # 1. Resumo executivo
    media_mercado = bench["sgs_25471_juros_veiculos_pf"]["ultimo_valor"]
    comparacao_mercado = (
        [S(f", acima da média do mercado, de {num(media_mercado)}%, e abaixo das instituições mais caras", novo=3)]
        if versao >= 3 else []
    )
    preencher(P[4], [
        S("Propomos aprovar as propostas dos scores 4 a 10 e negar os scores 1 a 3. A política combina taxa "
          "entre 2,3% e 2,9% ao mês"),
        S(" (2,54% em média", novo=True),
        *comparacao_mercado,
        S(")", novo=True),
        S(", "),
        S("o prazo que o cliente pediu, limitado a 60 meses,", novo=True),
        S(" e entrada mínima de 10%. No cenário-base, a projeção é de ROI anualizado de 16,4%, volume originado de "
          "R$ 52,6 milhões e inadimplência de 6,5%, com aprovação de 44,8% das propostas. Mesmo no cenário severo, "
          "a política permanece dentro dos limites: R$ 41,2 milhões de volume, 7,5% de inadimplência e ROI de "
          "16,0%. A proposta não busca extrair o maior retorno possível a qualquer custo; busca uma carteira "
          "rentável, com espaço para crescer e sem ultrapassar o apetite de risco definido pelo conselho."),
    ], destacar)
    apagar(P[5])  # o template deixa dois parágrafos vazios aqui; um basta

    # 2. O modelo de PD
    trocar_texto(P[9], "O que fizemos")
    celulas_modelo = {
        11: [S("LightGBM monotônico, com 200 árvores, 15 folhas por árvore, mínimo de 100 contratos por folha e "
               "taxa de aprendizado de 0,02.")],
        13: [S("As variáveis que mais contribuíram para a ordenação do risco foram idade do cliente, prazo, score "
               "de bureau, comprometimento de renda e tempo de emprego. Foram consideradas apenas informações "
               "disponíveis no momento da concessão.")],
        15: [S("A variável qtd_parcelas_em_atraso_12m foi descartada por vazamento: ela é conhecida depois da "
               "concessão, apresentou AUC isolada de 0,96 e vem zerada nas Bases B e C. A taxa antiga também foi "
               "excluída, pois não melhorou o resultado do modelo e não é uma variável neutra para construir a "
               "nova política."),
             S(" O ano_modelo saiu por repetir a idade do veículo e mudar com o calendário, o que distorce a "
               "validação no tempo. O alvo e os valores realizados (mês do default, EAD, LGD e perda) também "
               "ficaram de fora, porque só existem depois da concessão.", novo=True)],
        17: [S("Ausências em bureau, renda e tempo de emprego foram tratadas como informação. Criamos indicadores "
               "de ausência antes da imputação, porque a falta de dado pode sinalizar um perfil de risco diferente.")],
        19: [S("Usamos validação temporal: treino em 2022 e teste em 2023; depois treino em 2022–2023 e teste em "
               "2024. Não usamos split aleatório, porque ele mistura safras e tende a deixar a performance mais "
               "otimista do que seria em produção.")],
        21: [S("Treino (2022–2023): 0,845. Validação: ", novo=True),
             S("0,724 no teste de 2023 e 0,750 no teste de 2024, com média out-of-time de 0,737. "),
             S("Teste: ", novo=True),
             S("A submissão da Base B foi gerada com 3.000 contratos; o AUC oficial dessa base será conhecido "
               "somente na apuração, porque o alvo não é disponibilizado ao grupo."),
             S(" A distância entre treino e validação é esperada em modelos de árvore, que sempre ajustam melhor "
               "os dados em que aprenderam. Por isso o modelo foi escolhido pelo resultado fora do tempo, e a "
               "configuração vencedora é a mais contida.", novo=True)],
        23: [S("KS de 0,404 em 2024.")],
    }
    for i, segmentos in celulas_modelo.items():
        preencher(P[i], segmentos, destacar, tamanho=TAM_TABELA, alinhamento="left")

    logistica = next(c for c in modelo["campeao"]["candidatos"] if c["candidato"] == "logistica")
    P[24].addprevious(paragrafo_como(P[24], [S("")], destacar))  # respiro entre a tabela e o texto
    preencher(P[24], [
        S("O LightGBM monotônico foi escolhido por equilíbrio, não apenas pelo maior número. O ensemble teve AUC "
          "médio de 0,738, só 0,0014 acima do modelo escolhido; essa diferença não justificava adicionar "
          "complexidade. O modelo monotônico ficou muito próximo no AUC médio, teve melhor resultado em 2024 e "
          "apresentou KS de 0,404, acima do ensemble, que teve 0,381."),
        S(f" A regressão logística, mais simples, ficou em {logistica['auc_medio']:.3f}".replace(".", ",")
          + " de AUC médio, porque não captura relações que não são lineares.", novo=True),
    ], destacar)
    inserir_depois(P[24], [paragrafo_como(P[24], [
        S("A parte monotônica é uma proteção de bom senso. O modelo não pode aprender, por exemplo, que maior LTV, "
          "mais restrições, mais consultas ou maior comprometimento de renda reduzem a PD. Da mesma forma, não "
          "pode concluir que score de bureau, renda ou tempo de emprego maiores aumentam o risco. Isso reduz a "
          "chance de o modelo transformar uma coincidência da amostra histórica em uma regra de crédito. Ao mesmo "
          "tempo, ele continua capturando relações que não são lineares, como o comportamento em U da idade e os "
          "degraus de risco por prazo."),
        S(" Uma segunda implementação, feita de forma independente por outro integrante do grupo, confirmou o "
          "ganho das restrições monotônicas em 6 de 6 comparações.", novo=True),
    ], destacar)])

    # 3. Construção do score
    preencher(P[26], [
        S("Transformamos a PD em dez faixas por decis da distribuição da Base B, que é a carteira aprovada mais "
          "recente. Na prática, cada score reúne aproximadamente 10% da Base B. O score 1 representa a maior PD e "
          "o score 10 a menor."),
    ], destacar)
    inserir_depois(P[26], [
        paragrafo_como(P[26], [
            S("A escolha por decis torna as faixas mais fáceis de comparar e evita criar grupos muito pequenos com "
              "pouca estabilidade. Também ajuda a transformar uma probabilidade contínua em uma linguagem de "
              "política: em vez de discutir milhares de PDs individuais, definimos regras por dez faixas de risco."),
            S(" Os nove cortes são fixos: saíram uma vez da Base B e são aplicados por valor à Base C. Por isso a "
              "Base C não fica com 10% em cada faixa (36% das propostas caem no score 1), e a régua continua a "
              "mesma entre as bases.", novo=True),
        ], destacar),
        paragrafo_como(P[26], [
            S("A ordenação é coerente: a PD média prevista cai de 24,9% no score 1 para 2,5% no score 10; a perda "
              "esperada cai de 18,0% para 1,6% do valor financiado."),
            S(" No teste fora do tempo, a inadimplência observada em 2024 foi de 24,0% no score 1 contra 2,4% no "
              "score 10.", novo=True),
            S(" Em 2024, a inclinação de calibração foi de 1,08, dentro da faixa de referência de 0,8 a 1,2. Não "
              "houve necessidade de recalibrar o modelo."),
            S(" No mesmo teste, o modelo previu 8,6% de inadimplência média e ocorreram 7,2%: nos perfis que "
              "conhece, ele erra para o lado conservador.", novo=True),
        ], destacar),
        paragrafo_como(P[26], [
            S("Há uma ressalva importante para a política: a Base C é mais arriscada que a Base B. A PD média sobe "
              "de 7,8% para 14,4%, e 39,5% das propostas estão fora do perfil histórico conhecido pelo modelo. Por "
              "isso, nas simulações, aplicamos uma margem de segurança multiplicando por 2,0 a PD dos casos fora do "
              "perfil. Não é uma verdade observada; é uma hipótese prudencial para não tratar extrapolação como se "
              "fosse certeza."),
        ], destacar),
    ])

    # 4. A tabela de política (vem dos arquivos de resultado)
    apagar(P[29])
    regras = {r["score_1a10"]: r for r in politica["regras"]}
    limites = [None, *cortes, None]  # índice k: limite inferior do score (11 - k)
    for linha, score in enumerate(range(10, 0, -1)):
        base_i = 38 + 8 * linha
        inferior, superior = limites[linha], limites[linha + 1]
        if inferior is None:
            faixa = f"abaixo de {pct(superior, 2)}"
        elif superior is None:
            faixa = f"{pct(inferior, 2)} ou mais"
        else:
            faixa = f"{pct(inferior, 2)} a {pct(superior, 2)}"
        r = regras[score]
        aprova = bool(r["aprovar"])
        valores = [
            [S(faixa, novo=True)],
            [S("Aprovar" if aprova else "Negar")],
            [S(pct(r["taxa_am"]) if aprova else "—")],
            [S(f"até {int(r['prazo_max'])} m", novo=True)] if aprova else [S("—")],
            [S(pct(r["entrada_min"], 0) if aprova else "—")],
            [S(pct(faixas.loc[score, "perda_esperada_pct_media"]))],
            [S(pct(r["roi_base"]) if aprova else "—")],
        ]
        for coluna, segmentos in enumerate(valores, start=1):
            preencher(P[base_i + coluna], segmentos, destacar, tamanho=TAM_TABELA)
    preencher(P[119], [
        S("* ROI esperado por faixa no cenário-base da simulação. A perda esperada é calculada por PD × EAD × LGD, "
          "em percentual do valor financiado."),
        S(" Faixas de PD: o limite inferior pertence à faixa, e uma PD igual a um corte vai para a faixa de maior "
          "risco. Prazo: cada cliente recebe o prazo que pediu, limitado a 60 meses; a política nunca alonga um "
          "prazo.", novo=True),
    ], destacar, tamanho=17, cor="595959")

    # 5. Racional da precificação
    apagar(P[121])
    apagar(P[122])
    celulas_alavancas = {
        127: [S("Aprovar scores 4 a 10; negar scores 1 a 3.")],
        128: [S("Os scores 1 a 3 concentram a maior perda esperada: 18,0%, 8,6% e 6,3%. Aprovar a partir do score 4 "
                "mantém o risco dentro do limite e ainda preserva volume suficiente para o negócio."),
              S(" Abrir também o score 3 levaria a inadimplência do cenário severo a 9,1%, acima do limite.",
                novo=True)],
        130: [S("Taxas de 2,3% a 2,9% ao mês, aumentando com o risco.")],
        131: [S("A taxa precisa cobrir a perda esperada e deixar margem. A progressão de preço faz isso sem usar o "
                "teto de 3,5% de forma automática. Cobrar demais reduz aceite e pode piorar a carteira por seleção "
                "adversa.")],
        133: [S("Prazo pedido pelo cliente, limitado a 60 meses, em todas as faixas aprovadas.", novo=True)],
        134: [S("Encurtar prazo aumentou a parcela e o comprometimento de renda. Na simulação, isso tirou volume e "
                "ROI sem reduzir inadimplência o suficiente para compensar."),
              S(" A política nunca alonga um prazo. O risco maior de quem pede 60 meses já está no preço: o prazo é "
                "a segunda variável mais importante do modelo, então esses clientes caem em scores piores e pagam "
                "taxas maiores.", novo=True)],
        136: [S("10% em todas as faixas aprovadas.")],
        137: [S("A entrada reduz o LTV e, com ele, a LGD pela tabela de parâmetros. No nosso modelo o efeito direto "
                "na PD é pequeno, mas o enunciado indica que o cliente real pode reagir mais ao LTV.", novo=True),
              S(" É uma proteção simples sem exigir uma entrada tão alta que inviabilize o aceite e o volume.")],
    }
    for i, segmentos in celulas_alavancas.items():
        preencher(P[i], segmentos, destacar, tamanho=TAM_TABELA, alinhamento="left")
    modelo_texto = P[24]
    tabela_alavancas = tabela_de(P[126])
    inserir_depois(tabela_alavancas, [
        paragrafo_como(modelo_texto, [S("")], destacar),
        paragrafo_como(modelo_texto, [
            S("A taxa foi tratada como alavanca de equilíbrio, não como uma forma de maximizar receita no curto "
              "prazo. No cenário-base, as faixas aprovadas apresentam ROI entre 15,4% e 17,5%."),
            S(" Como esse ROI já desconta a perda esperada, ficar acima de 15% em todas as faixas aprovadas mostra "
              "que a taxa de cada uma cobre o próprio risco e ainda deixa margem, inclusive no score 4, o de maior "
              "risco (perda esperada de 4,9% e ROI de 17,2%).", novo=True),
            S(" A taxa mais alta fica no score 4, que tem maior risco entre os aprovados, mas ainda mantém conversão "
              "e retorno dentro do desenho da política."),
        ], destacar),
        paragrafo_como(modelo_texto, [
            S("Como teste de robustez, simulamos um aumento de 0,3 ponto percentual em todas as taxas. O ROI do "
              "cenário-base subiria para 18,6%, mas no cenário severo a inadimplência iria a 8,4% e o volume cairia "
              "para R$ 35,5 milhões. Ou seja: ganharíamos retorno no papel e perderíamos a política na prática, "
              "porque romperíamos dois guard-rails. Por isso, a proposta atual é mais equilibrada."),
        ], destacar),
    ])

    # 6. Resultado projetado
    apagar(P[139])
    celulas_resultado = {
        144: [S("44,8%")],
        147: [S("R$ 52,6 milhões no cenário-base; R$ 41,2 milhões no severo")],
        150: [S("6,5% no cenário-base; 7,5% no severo")],
        153: [S("2,54% a.m. em média (de 2,3% a 2,9% por faixa)", novo=True)],
        156: [S("16,4% no cenário-base; 16,0% no severo")],
    }
    for i, segmentos in celulas_resultado.items():
        preencher(P[i], segmentos, destacar, tamanho=TAM_TABELA)
    inserir_depois(tabela_de(P[143]), [
        paragrafo_como(modelo_texto, [S("")], destacar),
        paragrafo_como(modelo_texto, [
            S("A proposta foi analisada em três cenários de reação do cliente"),
            S(" (quantos aprovados aceitam a oferta e quanto a PD sobe com o preço)", novo=True),
            S(". No cenário brando, projeta 1.740 contratos, R$ 64,4 milhões de volume, 5,9% de inadimplência e "
              "ROI de 16,6%. No cenário-base, projeta 1.420 contratos, R$ 52,6 milhões, 6,5% de inadimplência e ROI "
              "de 16,4%. No cenário severo, projeta 1.110 contratos, R$ 41,2 milhões, 7,5% de inadimplência e ROI "
              "de 16,0%."),
        ], destacar),
        paragrafo_como(modelo_texto, [
            S("O cenário severo é o teste mais importante. Ele mostra que a política não depende de uma reação "
              "otimista do cliente para cumprir os limites. Ainda assim, o volume é o guard-rail mais apertado: a "
              "margem sobre o piso é de R$ 1,2 milhão. Isso exige disciplina na execução e acompanhamento próximo "
              "depois da implantação."),
        ], destacar),
    ])

    # 7. Riscos e limitações
    apagar(P[159])
    riscos = [
        [S("Inferência de rejeitados. ", negrito=True),
         S("As Bases A e B reúnem apenas contratos aprovados pela política antiga. A Base C é de mar aberto e inclui "
           "perfis que não aparecem no histórico de treino. Isso limita a precisão da PD para parte relevante da "
           "população; 39,5% das propostas estão fora do perfil histórico do modelo.")],
        [S("Mudança de composição da carteira. ", negrito=True),
         S("A PD média da Base C é 14,4%, contra 7,8% na Base B. A carteira nova é mais arriscada e tem forte "
           "concentração de propostas no score 1. Mesmo com margem de segurança, a projeção para esses perfis é "
           "mais incerta.")],
        [S("Seleção adversa e elasticidade de aceite. ", negrito=True),
         S("O modelo de PD não captura sozinho a mudança de perfil causada pelo preço. Taxas mais altas reduzem o "
           "aceite e podem aumentar a PD real dos contratos fechados. Por isso, os cenários são projeções e não "
           "garantias.")],
        [S("Deriva de safra. ", negrito=True),
         S("O desempenho de 2022 a 2024 não garante que a mesma relação entre variáveis e inadimplência continuará "
           "em 2025 e 2026."),
         S(" A inadimplência já oscilou no histórico: 8,9% em 2023 e 7,2% em 2024. Por isso validamos sempre no ano "
           "seguinte e mantivemos o nível da PD na média de 2022 a 2024, mais conservador.", novo=True),
         S(" A política precisa ser monitorada por safra, canal, score, LTV, prazo, entrada, faixa de taxa e status "
           "de fora do perfil.")],
        [S("Volume próximo ao limite no cenário severo. ", negrito=True),
         S("A política cumpre o mínimo de R$ 40 milhões, mas com folga reduzida. Qualquer piora adicional de aceite "
           "pode levar ao descumprimento do guard-rail de volume.")],
        [S("Sensibilidade ao nível da PD. ", novo=True, negrito=True),
         S("Testamos também um estresse de +20% na PD de toda a carteira. Nele, a inadimplência do cenário severo "
           "chegaria a 8,2%, acima do limite. Não adotamos esse estresse como premissa porque, no teste fora do "
           "tempo de 2024, o modelo superestimou a inadimplência dos perfis conhecidos (previu 8,6%, ocorreram "
           "7,2%). A incerteza que os dados não conseguem medir está nos perfis fora do histórico, e é neles que "
           "aplicamos a margem de segurança de 2×.", novo=True)],
        [S("Retomada e LGD elevada. ", negrito=True),
         S("A AutoCred tem operação de recuperação ainda imatura, com perdas após retomada próximas de 70%. Isso "
           "torna erros de concessão mais caros e reforça a necessidade de não relaxar a política apenas para "
           "aumentar volume.")],
    ]
    preencher(P[160], riscos[0], destacar)
    preencher(P[161], riscos[1], destacar)
    inserir_depois(P[161], [paragrafo_como(P[160], r, destacar) for r in riscos[2:]])
    preencher(P[162], [
        S("Em produção, a recomendação é acompanhar mensalmente a diferença entre PD prevista e inadimplência "
          "observada, a taxa de aceite, o volume, o LTV médio e a participação de casos fora do perfil. Se houver "
          "deterioração consistente, a primeira revisão deve ser de corte, entrada e precificação das faixas mais "
          "sensíveis, especialmente scores 4 e 5."),
    ], destacar)

    # 8. Divisão do trabalho (não estava no documento do Fábio)
    integrantes = {
        168: "Túlio Mazzarotto",
        170: "Leandro Nogueira",
        172: "Fábio Piona de Sousa (defesa) e Joyce Marques Barbosa (apoio e revisão)",
    }
    for i, nome in integrantes.items():
        preencher(P[i], [S(nome, novo=True)], destacar, tamanho=TAM_TABELA)
    if destacar == 2:
        for i in [164, 167, 169, 171]:  # o cabeçalho escuro da tabela fica sem grifo, para continuar legível
            grifar_existente(P[i])

    if versao >= 3:
        secao_benchmark(P, destacar, bench)
        for i, titulo in [(138, "7. Resultado projetado"), (158, "8. Riscos e limitações"),
                          (164, "9. Divisão do trabalho")]:
            trocar_texto(P[i], titulo)
            if destacar == 3:
                grifar_existente(P[i])

    return etree.tostring(raiz, xml_declaration=True, encoding="UTF-8", standalone=True)


def secao_benchmark(P, destacar, bench):
    """Seção 6 (v3): as taxas da política frente ao mercado de financiamento de veículos."""
    juros, inad, inst = (bench["sgs_25471_juros_veiculos_pf"], bench["sgs_21121_inadimplencia_veiculos_pf"],
                         bench["instituicoes"])
    grupos = inst["grupos"]
    antiga, media_periodo = bench["taxa_antiga_autocred"], juros["media_2022_2024"]
    media_politica = 2.54
    mais_baratas = sum(1 for t in bench["instituicoes_tipicas"] if t < media_politica)
    mes_recente, mes_inad = mes_ano(juros["ultimo_mes"]), mes_ano(inad["ultimo_mes"])
    faixa = lambda g: f"{num(grupos[g]['min'])}% a {num(grupos[g]['max'])}%"  # noqa: E731

    titulo = copia_sem_ids(P[120])
    trocar_texto(titulo, "6. Benchmarking com o mercado de financiamento de veículos")
    if destacar == 3:
        grifar_existente(titulo)

    texto = P[24]
    intro = paragrafo_como(texto, [S(
        "Para testar se as taxas propostas fazem sentido fora do nosso simulador, comparamos a política com dados "
        "públicos do Banco Central sobre o financiamento de veículos para pessoa física: a taxa média do mercado e "
        "a taxa praticada por cada instituição.", novo=3)], destacar)

    tabela = copia_sem_ids(next(a for a in P[143].iterancestors() if a.tag == w("tbl")))
    linhas = tabela.findall(w("tr"))
    cabecalho, modelos_linha = linhas[0], linhas[1:]
    for tr in modelos_linha:
        tabela.remove(tr)
    for tc, rotulo in zip(cabecalho.findall(w("tc")), ["Referência", "Taxa de juros (% a.m.)", "Leitura"]):
        trocar_texto(tc.find(w("p")), rotulo)
    conteudo = [
        (f"Média do mercado em 2022–2024 (período das Bases A e B)", f"{num(media_periodo)}%",
         "Referência do período em que a política antiga operou.", False),
        ("Taxa antiga da AutoCred (média da Base A)", f"{num(antiga)}%",
         f"{num(media_periodo - antiga)} ponto abaixo da média do mercado, para uma clientela mais arriscada.", False),
        (f"Média do mercado em {mes_recente}", f"{num(juros['ultimo_valor'])}%",
         "Referência mais recente, ponderada pelo volume concedido.", False),
        ("Bancos de montadoras (Mercedes-Benz, GM, Toyota, Volkswagen)", faixa("montadoras"),
         "Taxas promocionais para carro zero da própria marca. Não são comparáveis: a AutoCred financia sobretudo "
         "seminovos e usados, com idade média de 4 anos.", False),
        ("Grandes bancos (Caixa, Bradesco, Banco do Brasil, Santander, Itaú)", faixa("grandes_bancos"),
         "Onde o cliente de bom perfil tem alternativa.", False),
        (f"Mediana das {inst['quantidade']} instituições", f"{num(inst['mediana'])}%",
         "Metade do mercado cobra menos que isso.", False),
        ("Topo do mercado (Pan, Daycoval, Omni, Finamax)", faixa("topo"),
         "As taxas mais altas praticadas no período.", False),
        ("Política proposta da AutoCred", "2,3% a 2,9% (média de 2,54%)",
         f"Acima da média e da mediana do mercado, abaixo do topo. A média da política é maior que a taxa de "
         f"{mais_baratas} das {inst['quantidade']} instituições.", True),
    ]
    for k, (referencia, taxa, leitura, destaque_linha) in enumerate(conteudo):
        tr = copia_sem_ids(modelos_linha[k % 2])  # as duas primeiras linhas carregam a alternância de fundo
        celulas = [tc.find(w("p")) for tc in tr.findall(w("tc"))]
        preencher(celulas[0], [S(referencia, novo=3, negrito=destaque_linha)], destacar, tamanho=TAM_TABELA,
                  alinhamento="left")
        preencher(celulas[1], [S(taxa, novo=3, negrito=destaque_linha)], destacar, tamanho=TAM_TABELA,
                  alinhamento="center")
        preencher(celulas[2], [S(leitura, novo=3, negrito=destaque_linha)], destacar, tamanho=TAM_TABELA,
                  alinhamento="left")
        tabela.append(tr)
    definir_larguras(tabela, [0.40, 0.20, 0.40])

    periodo = " a ".join(f"{d[8:10]}/{d[5:7]}" for d in inst["periodo"])
    fonte = paragrafo_como(P[119], [S(
        f"Fontes: Banco Central do Brasil. SGS 25471 (taxa média mensal de juros, recursos livres, pessoas físicas, "
        f"aquisição de veículos); SGS 21121 (inadimplência da carteira, mesma modalidade); e Taxas de juros por "
        f"instituição financeira (Aquisição de veículos – Prefixado, pessoa física), usando a taxa típica de cada "
        f"instituição de {periodo}/2026. Consulta em {bench['consulta_em'][8:10]}/{bench['consulta_em'][5:7]}/"
        f"{bench['consulta_em'][:4]}. São taxas de juros, sem tarifas, IOF e seguros; o CET é maior.", novo=3)],
        destacar, tamanho=17, cor="595959")

    risco = paragrafo_como(texto, [
        S("O risco explica o prêmio. ", novo=3, negrito=True),
        S(f"A inadimplência (atraso acima de 90 dias) da carteira de veículos de pessoa física do sistema financeiro "
          f"ficou em {num(inad['media_2022_2024'], 1)}% em média de 2022 a 2024 e chegou a "
          f"{num(inad['ultimo_valor'], 1)}% em {mes_inad}. Na AutoCred, "
          f"{num(bench['inadimplencia_autocred'], 1)}% dos contratos de 2022 a 2024 atingiram 90 dias de atraso "
          f"em 12 meses, e a Base C projeta 14,4%. As métricas não são idênticas (a do Banco Central mede o estoque "
          f"da carteira; a nossa, cada safra ao longo de 12 meses), mas apontam na mesma direção: a AutoCred opera "
          f"com risco acima da média e não pode cobrar a taxa média.", novo=3),
    ], destacar)

    conclusoes = [
        [S("A política antiga cobrava abaixo do mercado ", novo=3, negrito=True),
         S(f"({num(antiga)}% contra {num(media_periodo)}%) por um risco acima do mercado. Preço baixo para risco "
           f"alto é parte da explicação da perda que o conselho identificou.", novo=3)],
        [S("A nova escada fica entre os grandes bancos e o topo. ", novo=3, negrito=True),
         S(f"A média de 2,54% está acima da média do mercado ({num(juros['ultimo_valor'])}%) e abaixo do topo "
           f"({faixa('topo')}). É o lugar coerente para uma carteira que nega os scores 1 a 3: mais risco que o "
           f"cliente de grande banco, menos que o das instituições mais caras.", novo=3)],
        [S("As melhores faixas ficam perto dos grandes bancos. ", novo=3, negrito=True),
         S(f"Nos scores 8 a 10, a taxa de 2,3% a 2,4% fica próxima do intervalo dos grandes bancos "
           f"({faixa('grandes_bancos')}), onde o bom cliente tem alternativa. Cobrar muito acima disso afastaria "
           f"justamente os clientes de menor risco: é a seleção adversa descrita no enunciado.", novo=3)],
        [S("O teto não precisa ser usado. ", novo=3, negrito=True),
         S(f"A taxa mais alta da política, 2,9% no score 4, fica no piso do grupo mais caro do mercado e bem abaixo "
           f"do teto de 3,5% a.m. definido pelo conselho.", novo=3)],
    ]
    lista = [paragrafo_como(P[160], c, destacar) for c in conclusoes]

    espaco = paragrafo_como(texto, [S("")], destacar)
    for elemento in [titulo, intro, tabela, fonte, risco, *lista, espaco]:
        P[138].addprevious(elemento)


def gravar_docx(document_xml: bytes, destino: Path) -> None:
    with zipfile.ZipFile(TEMPLATE) as origem, zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as saida:
        for item in origem.infolist():
            conteudo = document_xml if item.filename == "word/document.xml" else origem.read(item.filename)
            saida.writestr(item, conteudo)


def para_pdf(docx: Path) -> None:
    subprocess.run(
        [str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
        check=True, capture_output=True,
    )


def main() -> None:
    modelo, politica, faixas, cortes, bench = carregar_numeros()
    conferir(modelo, politica, faixas)
    DESTINO.mkdir(exist_ok=True)
    for versao, nome in VERSOES.items():
        for destacar, sufixo in [(None, ""), (versao, "_alteracoes_destacadas")]:
            docx = DESTINO / f"{nome}{sufixo}.docx"
            gravar_docx(montar(versao, destacar, modelo, politica, faixas, cortes, bench), docx)
            para_pdf(docx)
            print("gerado:", docx.name, "+ .pdf")


if __name__ == "__main__":
    main()
