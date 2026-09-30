# Política de crédito na Base C — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** construir o simulador de política com cenários de reação do cliente, conduzir o usuário pelas sete decisões e gravar a política escolhida (tabela por faixa e CSV no formato do exemplo).

**Architecture:** o motor puro e testado fica em `src/autocred/politica.py` e reaproveita `features.oferta_no_formato_a`, `escoragem.escorar` e as tabelas de EAD/LGD. As decisões são tomadas em conversa, com o `painel()` mostrando os três cenários. O notebook `04_politica` registra o caminho e grava as saídas.

**Tech Stack:** o mesmo do projeto (Python 3.14, pandas, numpy, pytest, jupytext/nbconvert).

**Spec:** `docs/specs/2026-09-19-politica-credito-design.md`

## Global Constraints

- Raiz: `modelo_pd/`. Python: `.venv/Scripts/python.exe`. Branch de trabalho: `politica-credito`.
- Limites do conselho: aprovação ≥ 0,35; inadimplência ≤ 0,08; volume contratado ≥ R$ 40.000.000; taxa ≤ 0,035 a.m. (truncada no teto). ROI alvo: > 0,15 a.a.
- Taxa de referência: `features.TAXA_REFERENCIA_AM` (0,01589). O score da proposta é o das condições pedidas, com os cortes de `saidas/modelo_pd.joblib`.
- Cenários: brando (0,90; 0,86; 0,05; 0,05; 0,10), base (0,85; 0,74; 0,10; 0,10; 0,22), severo (0,80; 0,61; 0,15; 0,15; 0,42), na ordem `aceite_base, fator_taxa, queda_entrada, queda_prazo, selecao_adversa`.
- Este exercício **não** é a submissão oficial do grupo.
- Commits com `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

| Arquivo | Responsabilidade |
|---|---|
| `src/autocred/politica.py` | cenários, juros Price, aceite, seleção adversa, regras, ofertas, simulação, painel, submissão e coerência |
| `tests/test_politica.py` | testes do motor (com modelo de mentira) |
| `notebooks/04_politica.py` | premissas, decisões, tabela final e saídas |

---

### Task 1: Motor `politica.py`

**Files:**
- Create: `src/autocred/politica.py`
- Test: `tests/test_politica.py`

**Interfaces:**
- Consumes: `escoragem.prever_pd`, `escoragem.escorar`, `features.oferta_no_formato_a`, `features.TAXA_REFERENCIA_AM`, `faixas.atribuir_score`, `perda.parcela_price`, `dados.PASTA_BASES`, `dados.ARQ_PARAMETROS`
- Produces:
  - `Cenario` (dataclass) e `CENARIOS`
  - constantes `TETO_TAXA_AM`, `APROVACAO_MIN`, `INADIMPLENCIA_MAX`, `VOLUME_MIN`, `ROI_ALVO`
  - `distribuicao_mes_default() -> np.ndarray`
  - `juros_ate(valor, taxa_am, prazo_meses, meses)`
  - `juros_esperados(valor, taxa_am, prazo_meses, prob_default) -> np.ndarray`
  - `aceite(cenario, taxa_am, delta_entrada, meses_cortados) -> np.ndarray`
  - `pd_ajustada(prob_default, cenario, taxa_am, fora_perfil=False, margem_fora_perfil=1.0) -> np.ndarray`
  - `montar_regras(score_minimo, taxa_am, prazo_max=60, entrada_min=0.0, aceita_fora_perfil=True) -> pd.DataFrame` (cada parâmetro aceita um escalar ou um dict {score: valor})
  - `preparar_base(modelo, propostas, cortes) -> pd.DataFrame` (+ `pd_referencia`, `score_1a10`, `fora_perfil`)
  - `ofertas(base, regras) -> pd.DataFrame`
  - `simular(modelo, base, cortes, regras, cenario, margem_fora_perfil=1.0) -> tuple[dict, pd.DataFrame]`
  - `resumir(detalhe, n_propostas) -> dict`
  - `painel(modelo, base, cortes, regras, margem_fora_perfil=1.0) -> pd.DataFrame`
  - `arquivo_submissao(base, regras) -> pd.DataFrame`
  - `validar_coerencia(submissao, base, regras, cortes) -> None`

- [ ] **Step 1: Escrever `tests/test_politica.py`**

```python
import numpy as np
import pandas as pd
import pytest

from autocred import perda, politica

CORTES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) / 2
BASE_CEN = politica.CENARIOS[1]


class _PDMetadeDoLTV:
    """Modelo de mentira: PD = LTV / 2."""

    def predict_proba(self, X):
        p = X["ltv"].to_numpy(dtype=float) / 2
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def base(base_c):
    return politica.preparar_base(_PDMetadeDoLTV(), base_c, CORTES)


def test_distribuicao_mes_default():
    d = politica.distribuicao_mes_default()
    assert len(d) == 12
    assert d.sum() == pytest.approx(1.0)


def test_juros_ate_primeira_parcela_e_total():
    pmt = perda.parcela_price(10_000, 0.02, 12)
    assert politica.juros_ate(10_000, 0.02, 12, 1) == pytest.approx(200.0)
    assert politica.juros_ate(10_000, 0.02, 12, 12) == pytest.approx(12 * pmt - 10_000)


def test_juros_esperados_extremos():
    total = politica.juros_ate(10_000, 0.02, 36, 36)
    dist = politica.distribuicao_mes_default()
    com_default = sum(p * politica.juros_ate(10_000, 0.02, 36, m) for m, p in enumerate(dist, start=1))
    assert politica.juros_esperados(10_000, 0.02, 36, 0.0)[0] == pytest.approx(total)
    assert politica.juros_esperados(10_000, 0.02, 36, 1.0)[0] == pytest.approx(com_default)


def test_aceite_na_referencia_e_efeitos():
    ref = politica.TAXA_REFERENCIA_AM
    assert politica.aceite(BASE_CEN, ref, 0, 0) == pytest.approx(0.85)
    assert politica.aceite(BASE_CEN, ref + 0.01, 0, 0) == pytest.approx(0.85 * 0.74)
    assert politica.aceite(BASE_CEN, ref, 0.10, 0) == pytest.approx(0.85 * 0.90)
    assert politica.aceite(BASE_CEN, ref, 0, 12) == pytest.approx(0.85 * 0.90)
    assert politica.aceite(BASE_CEN, 0.010, 0, 0) == pytest.approx(0.85)


def test_pd_ajustada():
    ref = politica.TAXA_REFERENCIA_AM
    assert politica.pd_ajustada(0.10, BASE_CEN, ref) == pytest.approx(0.10)
    assert politica.pd_ajustada(0.10, BASE_CEN, ref + 0.01) == pytest.approx(0.122)
    assert politica.pd_ajustada(0.10, BASE_CEN, ref, fora_perfil=True, margem_fora_perfil=1.5) == pytest.approx(0.15)
    assert politica.pd_ajustada(0.90, BASE_CEN, 0.035) == pytest.approx(0.99)


def test_montar_regras():
    r = politica.montar_regras(5, {s: 0.02 + s / 1000 for s in range(1, 11)}, prazo_max=48)
    assert list(r.index) == list(range(1, 11))
    assert r.loc[4, "aprovar"] == False and r.loc[5, "aprovar"] == True  # noqa: E712
    assert r.loc[10, "taxa_am"] == pytest.approx(0.03)
    assert (r["prazo_max"] == 48).all()


def test_ofertas_aplica_regras(base):
    regras = politica.montar_regras(5, 0.04, prazo_max=36, entrada_min=0.20, aceita_fora_perfil=False)
    of = politica.ofertas(base, regras)
    esperado_aprovar = (base["score_1a10"] >= 5) & ~base["fora_perfil"]
    assert (of["decisao"].eq("APROVAR") == esperado_aprovar).all()
    assert (of["taxa_am"] == 0.035).all()
    assert (of["prazo_meses"] == np.minimum(base["prazo_desejado_meses"], 36)).all()
    np.testing.assert_allclose(of["delta_entrada"], np.maximum(0, 0.20 - base["pct_entrada_desejada"]))


def test_simular_metricas_consistentes(base):
    regras = politica.montar_regras(3, 0.025, prazo_max=48, entrada_min=0.10)
    m, det = politica.simular(_PDMetadeDoLTV(), base, CORTES, regras, BASE_CEN)
    w = det["aceite"]
    assert m["aprovacao"] == pytest.approx(len(det) / len(base))
    assert m["volume"] == pytest.approx((w * det["valor_financiado"]).sum())
    prazo_anos = (w * det["prazo_meses"]).sum() / w.sum() / 12
    roi = ((w * det["juros_esperados"]).sum() - (w * det["perda_esperada"]).sum()) / m["volume"] / prazo_anos
    assert m["roi"] == pytest.approx(roi)
    assert m["ok_volume"] == (m["volume"] >= politica.VOLUME_MIN)


def test_painel_cenarios_ordenados(base):
    regras = politica.montar_regras(3, 0.03)
    p = politica.painel(_PDMetadeDoLTV(), base, CORTES, regras)
    assert list(p.index) == ["brando", "base", "severo"]
    assert p.loc["brando", "volume"] > p.loc["base", "volume"] > p.loc["severo", "volume"]
    assert p.loc["brando", "inadimplencia"] < p.loc["severo", "inadimplencia"]
    assert "ok_limites" in p.columns


def test_submissao_coerente_e_trava(base):
    regras = politica.montar_regras(5, 0.03, prazo_max=48, entrada_min=0.15)
    sub = politica.arquivo_submissao(base, regras)
    assert list(sub.columns) == ["id_proposta", "pd", "score_1a10", "decisao", "taxa_am", "prazo_meses", "pct_entrada_minima"]
    assert sub.loc[sub["decisao"] == "NEGAR", ["taxa_am", "prazo_meses", "pct_entrada_minima"]].isna().all().all()
    politica.validar_coerencia(sub, base, regras, CORTES)
    adulterada = sub.copy()
    linha = adulterada.index[adulterada["decisao"] == "APROVAR"][0]
    adulterada.loc[linha, "taxa_am"] = 0.02
    with pytest.raises(ValueError, match="taxa_am"):
        politica.validar_coerencia(adulterada, base, regras, CORTES)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_politica.py -q`
Expected: erro de coleta (`cannot import name 'politica'`)

- [ ] **Step 3: Implementar `src/autocred/politica.py`**

```python
"""Simulador de política de crédito na Base C: ofertas, reação do cliente e resultado da carteira."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from autocred import escoragem
from autocred.dados import ARQ_PARAMETROS, PASTA_BASES
from autocred.faixas import atribuir_score
from autocred.features import TAXA_REFERENCIA_AM, oferta_no_formato_a
from autocred.perda import parcela_price

TETO_TAXA_AM = 0.035
APROVACAO_MIN = 0.35
INADIMPLENCIA_MAX = 0.08
VOLUME_MIN = 40_000_000.0
ROI_ALVO = 0.15


@dataclass(frozen=True)
class Cenario:
    """Premissas de reação do cliente. Os efeitos são por +1 p.p. a.m. de taxa acima da referência,
    por +10 p.p. de entrada exigida além da desejada e por 12 meses de prazo cortados."""

    nome: str
    aceite_base: float
    fator_taxa: float
    queda_entrada: float
    queda_prazo: float
    selecao_adversa: float


CENARIOS = (
    Cenario("brando", 0.90, 0.86, 0.05, 0.05, 0.10),
    Cenario("base", 0.85, 0.74, 0.10, 0.10, 0.22),
    Cenario("severo", 0.80, 0.61, 0.15, 0.15, 0.42),
)


@lru_cache(maxsize=1)
def distribuicao_mes_default() -> np.ndarray:
    """Probabilidade de o calote cair em cada mês 1–12 (aba do arquivo de parâmetros)."""
    tabela = pd.read_excel(PASTA_BASES / ARQ_PARAMETROS, sheet_name="Distribuicao_Mes_Default")
    p = tabela.iloc[:, 1].to_numpy(dtype=float)
    return p / p.sum()


def _delta_taxa_pp(taxa_am) -> np.ndarray:
    return np.maximum(0.0, (np.asarray(taxa_am, dtype=float) - TAXA_REFERENCIA_AM) * 100)


def juros_ate(valor, taxa_am, prazo_meses, meses):
    """Juros pagos nas primeiras `meses` parcelas da Tabela Price: parcelas pagas − principal amortizado."""
    pv = np.asarray(valor, dtype=float)
    i = np.asarray(taxa_am, dtype=float)
    n = np.asarray(prazo_meses, dtype=float)
    m = np.asarray(meses, dtype=float)
    pmt = parcela_price(pv, i, n)
    saldo = pv * (1 + i) ** m - pmt * ((1 + i) ** m - 1) / i
    return m * pmt - (pv - saldo)


def juros_esperados(valor, taxa_am, prazo_meses, prob_default) -> np.ndarray:
    """(1 − PD) × juros do prazo inteiro + PD × juros esperados até o mês do calote."""
    pv = np.atleast_1d(np.asarray(valor, dtype=float))
    i = np.atleast_1d(np.asarray(taxa_am, dtype=float))
    n = np.atleast_1d(np.asarray(prazo_meses, dtype=float))
    p = np.atleast_1d(np.asarray(prob_default, dtype=float))
    dist = distribuicao_mes_default()
    meses = np.arange(1, len(dist) + 1, dtype=float)
    com_default = juros_ate(pv[:, None], i[:, None], n[:, None], meses[None, :]) @ dist
    sem_default = juros_ate(pv, i, n, n)
    return (1 - p) * sem_default + p * com_default


def aceite(cenario: Cenario, taxa_am, delta_entrada, meses_cortados) -> np.ndarray:
    """Chance de o cliente aceitar a oferta. Taxa abaixo da referência não aumenta o aceite."""
    a = cenario.aceite_base * cenario.fator_taxa ** _delta_taxa_pp(taxa_am)
    a = a * np.maximum(0.0, 1 - cenario.queda_entrada * np.asarray(delta_entrada, dtype=float) / 0.10)
    return a * np.maximum(0.0, 1 - cenario.queda_prazo * np.asarray(meses_cortados, dtype=float) / 12)


def pd_ajustada(prob_default, cenario: Cenario, taxa_am, fora_perfil=False, margem_fora_perfil: float = 1.0) -> np.ndarray:
    """PD do modelo × seleção adversa da taxa × margem para quem está fora do perfil (teto de 0,99)."""
    p = np.asarray(prob_default, dtype=float) * (1 + cenario.selecao_adversa) ** _delta_taxa_pp(taxa_am)
    p = p * np.where(np.asarray(fora_perfil), margem_fora_perfil, 1.0)
    return np.minimum(p, 0.99)


def _por_score(valor) -> list:
    if isinstance(valor, dict):
        return [valor[s] for s in range(1, 11)]
    return [valor] * 10


def montar_regras(score_minimo: int, taxa_am, prazo_max=60, entrada_min=0.0, aceita_fora_perfil=True) -> pd.DataFrame:
    """Tabela de regras por score. Cada parâmetro aceita um valor único ou um dict {score: valor}."""
    scores = range(1, 11)
    return pd.DataFrame(
        {
            "aprovar": [s >= score_minimo for s in scores],
            "taxa_am": _por_score(taxa_am),
            "prazo_max": _por_score(prazo_max),
            "entrada_min": _por_score(entrada_min),
            "aceita_fora_perfil": _por_score(aceita_fora_perfil),
        },
        index=pd.Index(scores, name="score_1a10"),
    )


def preparar_base(modelo, propostas: pd.DataFrame, cortes) -> pd.DataFrame:
    """Score e perfil de cada proposta nas condições pedidas (a oferta não muda a faixa)."""
    referencia = escoragem.prever_pd(modelo, propostas, cortes)
    base = propostas.copy()
    base["pd_referencia"] = referencia["pd"].to_numpy()
    base["score_1a10"] = referencia["score_1a10"].to_numpy()
    base["fora_perfil"] = referencia["fora_perfil_historico"].to_numpy()
    return base


def ofertas(base: pd.DataFrame, regras: pd.DataFrame) -> pd.DataFrame:
    """Aplica a tabela de regras a cada proposta: decisão e condições ofertadas."""
    r = regras.loc[base["score_1a10"].to_numpy()]
    fora = base["fora_perfil"].to_numpy(dtype=bool)
    aprovar = r["aprovar"].to_numpy(dtype=bool) & (~fora | r["aceita_fora_perfil"].to_numpy(dtype=bool))
    prazo_desejado = base["prazo_desejado_meses"].to_numpy()
    prazo = np.minimum(prazo_desejado, r["prazo_max"].to_numpy())
    entrada_min = r["entrada_min"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "id_proposta": base["id_proposta"].to_numpy(),
            "score_1a10": base["score_1a10"].to_numpy(),
            "fora_perfil": fora,
            "decisao": np.where(aprovar, "APROVAR", "NEGAR"),
            "taxa_am": np.minimum(r["taxa_am"].to_numpy(dtype=float), TETO_TAXA_AM),
            "prazo_meses": prazo.astype(int),
            "pct_entrada_minima": entrada_min,
            "delta_entrada": np.maximum(0.0, entrada_min - base["pct_entrada_desejada"].to_numpy(dtype=float)),
            "meses_cortados": prazo_desejado - prazo,
        },
        index=base.index,
    )


def resumir(detalhe: pd.DataFrame, n_propostas: int) -> dict:
    """Métricas da carteira, ponderadas pela chance de aceite de cada aprovado."""
    w = detalhe["aceite"].to_numpy()
    contratos = float(w.sum())
    aprovacao = len(detalhe) / n_propostas
    if contratos == 0:
        return {"aprovacao": aprovacao, "contratos": 0.0, "volume": 0.0, "inadimplencia": 0.0,
                "prazo_medio_anos": 0.0, "roi": float("nan"),
                "ok_aprovacao": aprovacao >= APROVACAO_MIN, "ok_inadimplencia": True, "ok_volume": False}
    volume = float((w * detalhe["valor_financiado"]).sum())
    inadimplencia = float((w * detalhe["pd_simulada"]).sum() / contratos)
    prazo_anos = float((w * detalhe["prazo_meses"]).sum() / contratos / 12)
    lucro = float((w * detalhe["juros_esperados"]).sum() - (w * detalhe["perda_esperada"]).sum())
    return {
        "aprovacao": aprovacao,
        "contratos": contratos,
        "volume": volume,
        "inadimplencia": inadimplencia,
        "prazo_medio_anos": prazo_anos,
        "roi": lucro / volume / prazo_anos,
        "ok_aprovacao": bool(aprovacao >= APROVACAO_MIN),
        "ok_inadimplencia": bool(inadimplencia <= INADIMPLENCIA_MAX),
        "ok_volume": bool(volume >= VOLUME_MIN),
    }


def simular(modelo, base: pd.DataFrame, cortes, regras: pd.DataFrame, cenario: Cenario,
            margem_fora_perfil: float = 1.0) -> tuple[dict, pd.DataFrame]:
    """Resultado esperado da carteira sob uma política e um cenário de reação do cliente."""
    of = ofertas(base, regras)
    aprovado = (of["decisao"] == "APROVAR").to_numpy()
    b = base.loc[aprovado]
    o = of.loc[aprovado]
    X = oferta_no_formato_a(b, o["taxa_am"].to_numpy(), o["prazo_meses"].to_numpy(), o["pct_entrada_minima"].to_numpy())
    esc = escoragem.escorar(modelo, X, cortes)
    taxa = o["taxa_am"].to_numpy()
    pd_sim = pd_ajustada(esc["pd"].to_numpy(), cenario, taxa, o["fora_perfil"].to_numpy(), margem_fora_perfil)
    detalhe = pd.DataFrame(
        {
            "id_proposta": o["id_proposta"].to_numpy(),
            "score_1a10": o["score_1a10"].to_numpy(),
            "aceite": aceite(cenario, taxa, o["delta_entrada"].to_numpy(), o["meses_cortados"].to_numpy()),
            "pd_simulada": pd_sim,
            "valor_financiado": X["valor_financiado"].to_numpy(dtype=float),
            "prazo_meses": X["prazo_meses"].to_numpy(dtype=float),
            "taxa_am": taxa,
            "juros_esperados": juros_esperados(X["valor_financiado"], taxa, X["prazo_meses"], pd_sim),
            "perda_esperada": pd_sim * esc["ead"].to_numpy() * esc["lgd"].to_numpy(),
        },
        index=b.index,
    )
    return resumir(detalhe, len(base)), detalhe


def painel(modelo, base: pd.DataFrame, cortes, regras: pd.DataFrame, margem_fora_perfil: float = 1.0) -> pd.DataFrame:
    """Uma linha por cenário, com as métricas e se cada limite do conselho foi cumprido."""
    linhas = []
    for cenario in CENARIOS:
        metricas, _ = simular(modelo, base, cortes, regras, cenario, margem_fora_perfil)
        linhas.append({"cenario": cenario.nome, **metricas})
    tabela = pd.DataFrame(linhas).set_index("cenario")
    tabela["ok_limites"] = tabela[["ok_aprovacao", "ok_inadimplencia", "ok_volume"]].all(axis=1)
    return tabela


def arquivo_submissao(base: pd.DataFrame, regras: pd.DataFrame) -> pd.DataFrame:
    """Formato do CSV de exemplo da política. Linhas NEGAR ficam sem taxa, prazo e entrada."""
    of = ofertas(base, regras)
    negar = (of["decisao"] == "NEGAR").to_numpy()
    sub = pd.DataFrame({
        "id_proposta": of["id_proposta"].to_numpy(),
        "pd": np.round(base["pd_referencia"].to_numpy(dtype=float), 6),
        "score_1a10": of["score_1a10"].to_numpy(),
        "decisao": of["decisao"].to_numpy(),
        "taxa_am": np.where(negar, np.nan, of["taxa_am"].to_numpy()),
        "prazo_meses": pd.array(np.where(negar, np.nan, of["prazo_meses"].to_numpy(dtype=float)), dtype="Int64"),
        "pct_entrada_minima": np.where(negar, np.nan, of["pct_entrada_minima"].to_numpy()),
    })
    return sub


def validar_coerencia(submissao: pd.DataFrame, base: pd.DataFrame, regras: pd.DataFrame, cortes) -> None:
    """Trava: cada linha submetida tem de seguir a regra da sua faixa (vale 10 pontos na avaliação)."""
    esperado = arquivo_submissao(base, regras)
    if len(submissao) != len(esperado):
        raise ValueError(f"esperava {len(esperado)} linhas, vieram {len(submissao)}")
    if not np.array_equal(submissao["id_proposta"].to_numpy(), esperado["id_proposta"].to_numpy()):
        raise ValueError("ids diferentes ou fora da ordem da Base C")
    if not np.array_equal(atribuir_score(submissao["pd"], cortes), submissao["score_1a10"].to_numpy()):
        raise ValueError("score_1a10 incoerente com a PD e os cortes")
    for coluna in ["decisao", "taxa_am", "prazo_meses", "pct_entrada_minima"]:
        a, b = submissao[coluna], esperado[coluna]
        iguais = (a.isna() & b.isna()) | (a == b)
        if not iguais.fillna(False).all():
            raise ValueError(f"coluna {coluna} diverge da tabela de regras")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest tests/test_politica.py -q` → Expected: `10 passed`. Depois, a suíte inteira: `78 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/autocred/politica.py tests/test_politica.py
git commit -m "feat: simulador de política com cenários de reação do cliente" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Sessão guiada: as sete decisões (em conversa)

**Exceção declarada:** o conteúdo desta tarefa são as escolhas do usuário, então não existe código final a fixar aqui. O que fica fixo é o procedimento de cada decisão:

- [ ] **Step 1: Carregar o contexto uma vez**

```python
import joblib
from autocred import dados, politica
pacote = joblib.load("saidas/modelo_pd.joblib")
modelo, cortes = pacote["modelo_portatil"], pacote["cortes"]
base = politica.preparar_base(modelo, dados.carregar_base("C"), cortes)
```

- [ ] **Step 2: Para cada decisão (1 premissas → 7 checagem):**
  - montar de 2 a 4 alternativas com `politica.montar_regras(...)`;
  - rodar `politica.painel(...)` em cada uma;
  - mostrar ao usuário aprovação, volume (R$ mi), inadimplência, ROI e `ok_limites` nos três cenários, com uma explicação do trade-off;
  - registrar a escolha e o porquê numa lista de decisões, que vai para o notebook.
- [ ] **Step 3:** repetir até a decisão 7. A política final precisa ter `ok_limites` verdadeiro no cenário **severo**.

---

### Task 3: Notebook `04_politica` e saídas

**Files:**
- Create: `notebooks/04_politica.py`
- Generated (fora do git): `saidas/tabela_politica.csv`, `saidas/politica_base_C.csv`

- [ ] **Step 1: Escrever o notebook.** Seções:
  1. premissas (tabela dos cenários);
  2. uma seção por decisão, com as alternativas comparadas, o `painel` de cada uma e a escolha com o porquê (texto da Task 2);
  3. a tabela final;
  4. o painel final;
  5. a gravação das saídas.
- [ ] **Step 2: Gravar e validar.**
  - `regras.to_csv("saidas/tabela_politica.csv")`
  - `sub = politica.arquivo_submissao(base, regras)`
  - `politica.validar_coerencia(sub, base, regras, cortes)`
  - `sub.to_csv("saidas/politica_base_C.csv", index=False)`
- [ ] **Step 3: Executar.**
  - `.venv/Scripts/jupytext --to ipynb notebooks/04_politica.py`
  - `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/04_politica.ipynb`
  - Expected: nenhuma célula com erro; `politica_base_C.csv` com 5.000 linhas + cabeçalho.
- [ ] **Step 4: Commit.** `git add notebooks/04_politica.py` + mensagem `feat: notebook da política de crédito na Base C` + Co-Authored-By.
