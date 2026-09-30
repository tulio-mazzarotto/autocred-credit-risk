"""Gera site/dados.json e site/og.png: todos os números da página-história do portfólio.

Lê os resultados do projeto (saidas/), o benchmarking (docs/benchmark/), os resultados oficiais
do grupo na apuração (docs/resultados_oficiais.json) e as bases, e pré-calcula a grade do simulador de
política (240 combinações × 3 cenários). Confere os números contra as fontes antes de gravar.
"""
import json
import math
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from autocred import dados, escoragem, modelos, politica  # noqa: E402
from autocred.avaliacao import auc, avaliar_temporal, mascaras_janela, psi  # noqa: E402
from autocred.faixas import atribuir_score  # noqa: E402

RAIZ = dados.RAIZ_PROJETO
SITE = RAIZ / "site"
ESCADA = {10: 0.023, 9: 0.023, 8: 0.024, 7: 0.025, 6: 0.026, 5: 0.028, 4: 0.029, 3: 0.031, 2: 0.035, 1: 0.035}
CORTES_SIM = [3, 4, 5, 6]
AJUSTES_PB = list(range(-30, 70, 10))  # −0,3 a +0,6 ponto percentual ao mês
ENTRADAS = [0.0, 0.10, 0.20]
FORA = ["aceitar", "negar"]
MARGEM_FORA_PERFIL = 2.0


def chave(corte: int, ajuste_pb: int, entrada: float, fora: str) -> str:
    return f"{corte}|{ajuste_pb}|{round(entrada * 100)}|{fora}"


VENCEDORA = chave(4, 0, 0.10, "aceitar")


def ler(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def limpar(obj):
    """JSON válido para o navegador: NaN vira null e os tipos do numpy viram nativos."""
    if isinstance(obj, dict):
        return {str(k): limpar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [limpar(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def links_da_pagina() -> dict:
    """Endereços do repositório e da página publicada.

    No projeto de trabalho vêm de publico/config.json; o repositório público não leva essa pasta,
    então lá os endereços continuam os do dados.json já gerado.
    """
    config = RAIZ / "publico" / "config.json"
    if not config.exists():
        return ler(SITE / "dados.json")["links"]
    publico = ler(config)
    usuario, repositorio = publico["usuario_github"], publico["repositorio"]
    return {"repositorio": f"https://github.com/{usuario}/{repositorio}",
            "pagina": f"https://{usuario}.github.io/{repositorio}/"}


def secao_caso(A, B, C) -> dict:
    anos = dados.ano(A)
    return {
        "contratos_A": len(A), "contratos_B": len(B), "propostas_C": len(C),
        "inadimplencia_A": float(A[dados.ALVO].mean()),
        "inadimplencia_por_ano": {int(a): float(v) for a, v in A.groupby(anos)[dados.ALVO].mean().items()},
        "ltv_medio": float(A["ltv"].mean()),
        "lgd_media_realizada": float(A["lgd_realizado"].mean()),
        "taxa_antiga": float(A["taxa_juros_am"].mean()),
    }


def secao_modelo(A, resultados: dict) -> dict:
    campeao = resultados["campeao"]
    y = A[dados.ALVO].to_numpy()
    a_vaz = auc(y, A["qtd_parcelas_em_atraso_12m"])
    faixas_idade = pd.cut(A["idade_cliente"], [20, 25, 30, 35, 40, 50, 60, 120], right=False,
                          labels=["20–24", "25–29", "30–34", "35–39", "40–49", "50–59", "60+"])
    idade = A.groupby(faixas_idade, observed=True)[dados.ALVO].mean()
    prazo = A.groupby("prazo_meses")[dados.ALVO].mean()
    anos = dados.ano(A)
    X = A[dados.colunas_entrada("portatil")]
    _, previsoes = avaliar_temporal(lambda: modelos.construir_modelo(campeao["descricao"]), X, y, anos)
    _, teste_2024 = mascaras_janela(anos, "J2")
    colunas_faixa = ["score_1a10", "pd_min", "pd_max", "pd_media", "inadimplencia_observada_J2",
                     "perda_esperada_pct_media", "qtd_base_C"]
    return {
        **{k: campeao[k] for k in ["auc_J1", "auc_J2", "auc_medio", "ks_J2", "ic95_auc_J2"]},
        "descricao": campeao["descricao"],
        "candidatos": campeao["candidatos"],
        "importancias": resultados["importancias"][:5],
        "auc_vazamento": max(a_vaz, 1 - a_vaz),
        "inadimplencia_por_idade": [{"faixa": str(k), "taxa": float(v)} for k, v in idade.items()],
        "inadimplencia_por_prazo": [{"prazo": int(k), "taxa": float(v)} for k, v in prazo.items()],
        "faixas": [{k: f[k] for k in colunas_faixa} for f in resultados["faixas"]],
        "cortes": resultados["cortes"],
        "pd_media_B": resultados["pd_media_B"],
        "pd_media_C": resultados["pd_media_C"],
        "pct_fora_perfil_C": resultados["pct_fora_perfil_C"],
        "inclinacao_J2": resultados["info_portatil"]["inclinacao_J2"],
        "previsto_2024": float(previsoes["J2"].mean()),
        "observado_2024": float(y[teste_2024].mean()),
    }


def secao_politica(resultados: dict) -> dict:
    return {
        "margem_fora_perfil": resultados["margem_fora_perfil"],
        "regras": resultados["regras"],
        "final": {linha["cenario"]: linha for linha in resultados["final"]},
        "agressiva": {linha["cenario"]: linha for linha in resultados["agressiva"]},
    }


def secao_simulador(modelo, cortes, base) -> dict:
    resultados = {}
    for corte in CORTES_SIM:
        for ajuste in AJUSTES_PB:
            taxa = {s: min(ESCADA[s] + ajuste / 10_000, politica.TETO_TAXA_AM) for s in range(1, 11)}
            for entrada in ENTRADAS:
                for fora in FORA:
                    regras = politica.montar_regras(corte, taxa, prazo_max=60, entrada_min=entrada,
                                                    aceita_fora_perfil=(fora == "aceitar"))
                    painel = politica.painel(modelo, base, cortes, regras, MARGEM_FORA_PERFIL)
                    resultados[chave(corte, ajuste, entrada, fora)] = {
                        cenario: {
                            "aprovacao": round(float(p["aprovacao"]), 4),
                            "contratos": round(float(p["contratos"]), 1),
                            "volume_mi": round(float(p["volume"]) / 1e6, 2),
                            "inadimplencia": round(float(p["inadimplencia"]), 4),
                            "roi": round(float(p["roi"]), 4),
                            "ok_aprovacao": bool(p["ok_aprovacao"]),
                            "ok_inadimplencia": bool(p["ok_inadimplencia"]),
                            "ok_volume": bool(p["ok_volume"]),
                        }
                        for cenario, p in painel.iterrows()
                    }
        print(f"simulador: corte {corte} pronto")
    return {"cortes": CORTES_SIM, "ajustes_pb": AJUSTES_PB, "entradas": ENTRADAS, "fora_perfil": FORA,
            "escada": ESCADA, "teto_taxa_am": politica.TETO_TAXA_AM, "vencedora": VENCEDORA,
            "resultados": resultados}


def psi_entre_safras(pacote: dict, A, ano_referencia: int, ano_atual: int) -> float:
    """PSI da distribuição dos scores 1–10 do modelo final entre duas safras da Base A."""
    scores = atribuir_score(escoragem.prever(pacote["modelo_portatil"], A), pacote["cortes"])
    anos = dados.ano(A).to_numpy()
    return psi(scores[anos == ano_referencia], scores[anos == ano_atual])


def secao_benchmark(A) -> dict:
    bcb = ler(RAIZ / "docs" / "benchmark" / "benchmark_bcb.json")
    instituicoes = pd.read_csv(RAIZ / "docs" / "benchmark" / "benchmark_instituicoes.csv")
    return {
        **bcb,
        "taxas_instituicoes": [
            {"instituicao": linha.InstituicaoFinanceira, "taxa": float(linha.taxa_tipica)}
            for linha in instituicoes.itertuples()
        ],
        "taxa_antiga_autocred": float(A["taxa_juros_am"].mean()) * 100,
        "politica_min": 2.3, "politica_max": 2.9, "politica_media": 2.54,
    }


def conferir(pagina: dict, resultados_modelo: dict, resultados_politica: dict) -> None:
    vencedora = pagina["simulador"]["resultados"][VENCEDORA]
    for linha in resultados_politica["final"]:
        c = linha["cenario"]
        assert abs(vencedora[c]["roi"] - linha["roi"]) < 1e-4, f"ROI do simulador diverge no cenário {c}"
        assert abs(vencedora[c]["volume_mi"] - linha["volume"] / 1e6) < 0.01, f"volume diverge no cenário {c}"
    assert pagina["modelo"]["auc_medio"] == resultados_modelo["campeao"]["auc_medio"]
    assert len(pagina["simulador"]["resultados"]) == 240


def gerar_og(oficial: dict) -> None:
    """Cartão de pré-visualização do LinkedIn (1200×630), na paleta da página: marinho e dourado."""
    marinho, texto, ouro, linha = "#0F172A", "#334155", "#B45309", "#E2E8F0"
    virgula = lambda valor: valor.replace(".", ",")  # noqa: E731
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor("#FAF9F6")
    fig.text(0.06, 0.87, "AUTOCRED  ·  ESTUDO DE CASO DE CRÉDITO  ·  CREDIT CASE STUDY", color=ouro, fontsize=14,
             fontweight="bold", family="DejaVu Sans")
    fig.text(0.06, 0.60, "Otimizando risco e precificação\nde crédito automotivo", color=marinho, fontsize=38,
             fontweight="bold", family="DejaVu Serif", linespacing=1.12)
    fig.text(0.06, 0.505, "Modelagem de PD, política de taxas e ROI realizado", color=texto, fontsize=17)
    fig.text(0.06, 0.45, "PD model, rate policy and realized ROI", color="#64748B", fontsize=15, style="italic")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.37, 0.37], color=linha, linewidth=1.5, transform=fig.transFigure))
    destaques = [
        (virgula(f"{oficial['auc']:.4f}"), "AUC fora do tempo · out-of-time AUC"),
        (virgula(f"{oficial['roi'] * 100:.2f}%"), f"ROI realizado · meta > {oficial['meta_roi']:.0%}"),
        (virgula(f"{oficial['inadimplencia'] * 100:.2f}%"), "inadimplência · default rate"),
    ]
    for i, (numero, rotulo) in enumerate(destaques):
        x = 0.06 + i * 0.31
        fig.text(x, 0.19, numero, color=ouro, fontsize=40, fontweight="bold", family="DejaVu Sans Mono")
        fig.text(x, 0.11, rotulo, color=texto, fontsize=13)
    fig.savefig(SITE / "og.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    SITE.mkdir(exist_ok=True)
    saidas = dados.PASTA_SAIDAS
    resultados_modelo = ler(saidas / "resultados_modelo.json")
    resultados_politica = ler(saidas / "resultados_politica.json")
    oficial = ler(RAIZ / "docs" / "resultados_oficiais.json")
    pacote = joblib.load(saidas / "modelo_pd.joblib")
    A, B, C = dados.carregar_base("A"), dados.carregar_base("B"), dados.carregar_base("C")
    base = politica.preparar_base(pacote["modelo_portatil"], C, pacote["cortes"])

    pagina = limpar({
        "gerado_em": date.today().isoformat(),
        "links": links_da_pagina(),
        "caso": secao_caso(A, B, C),
        "modelo": {**secao_modelo(A, resultados_modelo), "psi_2023_2024": psi_entre_safras(pacote, A, 2023, 2024)},
        "politica": secao_politica(resultados_politica),
        "simulador": secao_simulador(pacote["modelo_portatil"], pacote["cortes"], base),
        "benchmark": secao_benchmark(A),
        "limites": {"aprovacao_min": politica.APROVACAO_MIN, "inadimplencia_max": politica.INADIMPLENCIA_MAX,
                    "volume_min_mi": politica.VOLUME_MIN / 1e6, "taxa_max_am": politica.TETO_TAXA_AM},
        "oficial": oficial,
    })
    conferir(pagina, resultados_modelo, resultados_politica)
    with open(SITE / "dados.json", "w", encoding="utf-8") as arquivo:
        json.dump(pagina, arquivo, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    gerar_og(oficial)
    print("gravados: site/dados.json e site/og.png")


if __name__ == "__main__":
    main()
