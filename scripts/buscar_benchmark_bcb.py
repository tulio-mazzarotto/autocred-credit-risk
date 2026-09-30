"""Benchmarking de taxas: baixa dados públicos do Banco Central e grava docs/benchmark/.

Fontes (APIs públicas, sem autenticação):
  - SGS 25471: taxa média mensal de juros, recursos livres, PF, aquisição de veículos (% a.m.);
  - SGS 21121: inadimplência da carteira (atraso > 90 dias), recursos livres, PF, aquisição de veículos (%);
  - Olinda, taxaJuros v2: taxa de cada instituição, modalidade "Aquisição de veículos - Prefixado", PF.

Uso: python scripts/buscar_benchmark_bcb.py [inicio_AAAA-MM-DD]
"""
import json
import statistics as st
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from autocred import dados

DESTINO = dados.RAIZ_PROJETO / "docs" / "benchmark"
SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial=01/01/2022"
OLINDA = "https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata/TaxasJurosDiariaPorInicioPeriodo"

# Grupos citados no documento (nomes exatamente como o Banco Central publica).
GRUPOS = {
    "montadoras": ["BCO MERCEDES-BENZ S.A.", "BCO GM S.A.", "BCO TOYOTA DO BRASIL S.A.", "BCO VOLKSWAGEN S.A"],
    "grandes_bancos": ["CAIXA ECONOMICA FEDERAL", "BCO BRADESCO S.A.", "BCO DO BRASIL S.A.", "SANTANDER SCFI S.A.",
                       "ITAÚ UNIBANCO HOLDING S.A."],
    "topo": ["BANCO PAN", "BCO DAYCOVAL S.A", "OMNI SA CFI", "FINAMAX S.A. CFI"],
}


def baixar_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=180) as resposta:
        return json.load(resposta)


def serie_sgs(codigo: int) -> pd.Series:
    bruto = baixar_json(SGS.format(codigo=codigo))
    serie = pd.Series({pd.to_datetime(x["data"], dayfirst=True): float(x["valor"]) for x in bruto})
    return serie.sort_index()


def taxas_por_instituicao(inicio: str) -> pd.DataFrame:
    filtro = (f"Modalidade eq 'Aquisição de veículos - Prefixado' and Segmento eq 'PESSOA FÍSICA' "
              f"and InicioPeriodo ge '{inicio}'")
    url = f"{OLINDA}?%24format=json&%24top=20000&%24filter={urllib.parse.quote(filtro)}"
    return pd.DataFrame(baixar_json(url)["value"])


def main() -> None:
    inicio = sys.argv[1] if len(sys.argv) > 1 else "2026-07-01"
    DESTINO.mkdir(parents=True, exist_ok=True)

    juros = serie_sgs(25471)
    inad = serie_sgs(21121)
    periodo_bases = (juros.index.year >= 2022) & (juros.index.year <= 2024)
    periodo_inad = (inad.index.year >= 2022) & (inad.index.year <= 2024)

    diarias = taxas_por_instituicao(inicio)
    # Taxa típica de cada instituição: mediana dos períodos de 5 dias úteis em que ela aparece.
    tipicas = (diarias.groupby("InstituicaoFinanceira")["TaxaJurosAoMes"]
               .agg(taxa_tipica="median", periodos="size").sort_values("taxa_tipica"))
    tipicas.to_csv(DESTINO / "benchmark_instituicoes.csv", encoding="utf-8")
    t = tipicas["taxa_tipica"].tolist()
    quartis = st.quantiles(t, n=4)

    grupos = {}
    for nome, membros in GRUPOS.items():
        faltando = [m for m in membros if m not in tipicas.index]
        if faltando:
            raise ValueError(f"instituições ausentes no período: {faltando}")
        valores = tipicas.loc[membros, "taxa_tipica"]
        grupos[nome] = {"membros": membros, "min": float(valores.min()), "max": float(valores.max())}

    resultado = {
        "consulta_em": date.today().isoformat(),
        "sgs_25471_juros_veiculos_pf": {
            "media_2022_2024": round(float(juros[periodo_bases].mean()), 4),
            "ultimo_mes": juros.index[-1].strftime("%Y-%m"),
            "ultimo_valor": float(juros.iloc[-1]),
        },
        "sgs_21121_inadimplencia_veiculos_pf": {
            "media_2022_2024": round(float(inad[periodo_inad].mean()), 4),
            "ultimo_mes": inad.index[-1].strftime("%Y-%m"),
            "ultimo_valor": float(inad.iloc[-1]),
        },
        "instituicoes": {
            "periodo": [diarias["InicioPeriodo"].min(), diarias["FimPeriodo"].max()],
            "quantidade": len(t),
            "minima": min(t), "p25": quartis[0], "mediana": st.median(t), "p75": quartis[2],
            "p90": st.quantiles(t, n=10)[8], "maxima": max(t),
            "grupos": grupos,
        },
    }
    with open(DESTINO / "benchmark_bcb.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
