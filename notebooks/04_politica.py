# %% [markdown]
# # 04 · Política de crédito na Base C (exercício guiado)
#
# **Objetivo:** transformar o score em decisão (a quem conceder, a que taxa, em que prazo e com quanta entrada) e
# levar uma proposta para comparar com a do colega responsável pela política. **Não é a submissão oficial.**
#
# **Limites do conselho:** aprovação ≥ 35% das 5.000 propostas · inadimplência ≤ 8% dos contratos ·
# volume contratado ≥ R$ 40 mi · taxa ≤ 3,5% a.m. **Meta:** ROI anual acima de 15%.
#
# **ROI anual** = [(juros recebidos − perda) ÷ volume financiado] ÷ prazo médio em anos. Quem quebra paga juros
# só até o mês do calote; proposta aprovada que o cliente recusa não entra na conta.
#
# **O problema:** a Base C reage à oferta. Taxa alta afasta o cliente e atrai quem não tem alternativa
# (seleção adversa); entrada e prazo curto também derrubam o aceite. O enunciado dá a direção desses efeitos, mas
# não a intensidade. Por isso simulamos **três cenários** e exigimos que a política cumpra os limites mesmo no severo.

# %%
import json

import joblib
import pandas as pd

from autocred import dados, politica

pd.set_option("display.width", 220)
SAIDAS = dados.PASTA_SAIDAS
pacote = joblib.load(SAIDAS / "modelo_pd.joblib")
modelo, cortes = pacote["modelo_portatil"], pacote["cortes"]
base = politica.preparar_base(modelo, dados.carregar_base("C"), cortes)


def resumo(nome, regras, margem=1.0):
    p = politica.painel(modelo, base, cortes, regras, margem)
    return {
        "alternativa": nome,
        "aprov.": f"{p.aprovacao.iloc[0]:.1%}",
        "vol. base (R$ mi)": round(p.loc["base", "volume"] / 1e6, 1),
        "vol. severo (R$ mi)": round(p.loc["severo", "volume"] / 1e6, 1),
        "inad. base": f"{p.loc['base', 'inadimplencia']:.1%}",
        "inad. severo": f"{p.loc['severo', 'inadimplencia']:.1%}",
        "ROI base": f"{p.loc['base', 'roi']:.1%}",
        "ROI severo": f"{p.loc['severo', 'roi']:.1%}",
        "limites ok (brando/base/severo)": "/".join("sim" if v else "NÃO" for v in p.ok_limites),
    }


def comparar(alternativas, margem=1.0):
    display(pd.DataFrame([resumo(n, r, margem) for n, r in alternativas]))


# %% [markdown]
# ## 1. Premissas de reação do cliente
#
# A taxa de referência é a média antiga, 1,589% a.m. Os efeitos valem por +1 ponto de taxa ao mês acima dela,
# por +10 pontos de entrada exigida além da desejada e por 12 meses de prazo cortados.

# %%
display(pd.DataFrame([c.__dict__ for c in politica.CENARIOS]).set_index("nome"))
efeito = []
for taxa in [0.0159, 0.020, 0.025, 0.030, 0.035]:
    linha = {"taxa a.m.": f"{taxa:.2%}"}
    for c in politica.CENARIOS:
        aceite = politica.aceite(c, taxa, 0, 0) / c.aceite_base
        mult_pd = (1 + c.selecao_adversa) ** max(0.0, (taxa - politica.TAXA_REFERENCIA_AM) * 100)
        linha[c.nome] = f"aceite {aceite:.0%} · PD ×{mult_pd:.2f}"
    efeito.append(linha)
display(pd.DataFrame(efeito))

# %% [markdown]
# **Três políticas ingênuas, para calibrar a intuição:**

# %%
comparar([
    ("A) aprovar todos a 1,59%", politica.montar_regras(1, 0.0159)),
    ("B) scores 5–10 a 1,59%", politica.montar_regras(5, 0.0159)),
    ("C) scores 5–10 a 3,5%", politica.montar_regras(5, 0.035)),
])

# %% [markdown]
# - **A** repete o problema do caso: sem filtro, a inadimplência passa de 14% e o ROI fica em ~7%.
# - **B**: filtrar resolve o risco, mas o preço antigo não entrega os 15%.
# - **C**: cobrar o teto dá ROI alto, mas afasta os clientes (volume abaixo de R$ 40 mi) e, no severo, a seleção
#   adversa estoura a inadimplência. É a "política degenerada" do enunciado.
#
# **Decisão 1:** manter as premissas como estão. O severo já é duro: a 3,5% a.m., perde 61% dos clientes e quase dobra a PD.
#
# ## 2. Corte: quais scores aprovar (taxa provisória de 2,5% a.m.)

# %%
comparar([(f"scores {s}–10", politica.montar_regras(s, 0.025)) for s in [7, 6, 5, 4, 3, 2]])

# %% [markdown]
# **Decisão 2: aprovar os scores 4 a 10.** É o corte que cumpre os limites nos três cenários com folga dos dois lados:
# o 5–10 fura o volume no severo, e o 3–10 deixa a inadimplência do severo colada nos 8%.
#
# ## 3. Taxa por faixa
#
# Primeiro, a economia de cada faixa. Com a definição de ROI do enunciado, a perda (que acontece uma vez) é diluída
# pelos ~3,5 anos de prazo médio, então **o nível da taxa pesa muito mais que a faixa**.

# %%
for cenario in politica.CENARIOS[1:]:
    tabela = {}
    for taxa in [0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.035]:
        _, det = politica.simular(modelo, base, cortes, politica.montar_regras(4, taxa), cenario)
        w = det["aceite"]
        g = det.assign(
            wj=w * det["juros_esperados"], wp=w * det["perda_esperada"], wv=w * det["valor_financiado"],
            wz=w * det["prazo_meses"], w=w,
        ).groupby("score_1a10")[["wj", "wp", "wv", "wz", "w"]].sum()
        tabela[f"{taxa:.1%}"] = ((g["wj"] - g["wp"]) / g["wv"] / (g["wz"] / g["w"] / 12)).map("{:.0%}".format)
    print(f"ROI por faixa, cenário {cenario.nome}")
    display(pd.DataFrame(tabela).sort_index(ascending=False))

# %%
ESCADA = {1: 0.035, 2: 0.035, 3: 0.035, 4: 0.029, 5: 0.028, 6: 0.026, 7: 0.025, 8: 0.024, 9: 0.023, 10: 0.023}
comparar([
    ("plana 2,4%", politica.montar_regras(4, 0.024)),
    ("plana 2,5%", politica.montar_regras(4, 0.025)),
    ("plana 2,6%", politica.montar_regras(4, 0.026)),
    ("plana 2,7%", politica.montar_regras(4, 0.027)),
    ("escada por risco 2,3% → 2,9%", politica.montar_regras(4, ESCADA)),
])

# %% [markdown]
# Cada +0,1 ponto de taxa rende ~+0,8 ponto de ROI e custa ~R$ 2 mi de volume no severo. A 2,7% o volume já fura o piso.
#
# **Decisão 3: escada por risco.** Scores 10–9 pagam 2,3%; 8: 2,4%; 7: 2,5%; 6: 2,6%; 5: 2,8%; 4: 2,9% a.m. Rende o
# mesmo que a plana de 2,5% e conta a história certa para a defesa: a taxa sobe conforme a perda esperada da faixa.
#
# ## 4. Prazo máximo

# %%
def por_score(padrao, especificos):
    return {s: especificos.get(s, padrao) for s in range(1, 11)}


comparar([
    ("sem limite (60 m)", politica.montar_regras(4, ESCADA)),
    ("48 m para todos", politica.montar_regras(4, ESCADA, prazo_max=48)),
    ("48 m nos scores 4–6", politica.montar_regras(4, ESCADA, prazo_max=por_score(60, {4: 48, 5: 48, 6: 48}))),
    ("36 m nos 4–5, 48 m nos 6–7", politica.montar_regras(4, ESCADA, prazo_max=por_score(60, {4: 36, 5: 36, 6: 48, 7: 48}))),
])

# %% [markdown]
# Encurtar o prazo **não reduziu a inadimplência**. A parcela sobe e o comprometimento de renda (4ª variável mais
# importante do modelo) compensa o prazo menor. Só custou aceite, volume e ROI, porque a perda passa a ser
# diluída em menos anos.
#
# **Decisão 4: sem limite de prazo (60 meses para todos).**
#
# ## 5. Entrada mínima

# %%
comparar([
    ("sem mínimo", politica.montar_regras(4, ESCADA)),
    ("10% para todos", politica.montar_regras(4, ESCADA, entrada_min=0.10)),
    ("20% para todos", politica.montar_regras(4, ESCADA, entrada_min=0.20)),
    ("20% nos scores 4–5", politica.montar_regras(4, ESCADA, entrada_min=por_score(0.0, {4: 0.20, 5: 0.20}))),
])

# %% [markdown]
# No nosso modelo o LTV quase não pesa na PD, então o ganho vem pela LGD e é pequeno. Mas o enunciado diz que, no
# simulador, LTV menor reduz PD **e** LGD, então o efeito real pode ser maior. Com 10%, só 12% dos aprovados são
# afetados, e quem pede LTV acima de 95% volta ao patamar que a AutoCred já conhecia.
#
# **Decisão 5: entrada mínima de 10% para todas as faixas.** É um seguro barato: −0,2 ponto de inadimplência e só
# −R$ 0,4 mi de volume no severo.
#
# ## 6. Propostas fora do perfil histórico

# %%
def regras_com(aceita):
    return politica.montar_regras(4, ESCADA, entrada_min=0.10, aceita_fora_perfil=aceita)


aprovados = base[base["score_1a10"] >= 4]
print(f"Aprovados fora do perfil: {aprovados['fora_perfil'].sum()} de {len(aprovados)} ({aprovados['fora_perfil'].mean():.1%})")
negar_4_6 = por_score(True, {4: False, 5: False, 6: False})
comparar([
    ("aceitar (PD do modelo)", regras_com(True)),
    ("negar fora do perfil nos 4–6", regras_com(negar_4_6)),
    ("negar fora do perfil em todas", regras_com(False)),
])
print("Estresse: a mesma regra de aceitar, avaliada com a PD dos fora do perfil dobrada")
comparar([("aceitar, estresse PD ×2", regras_com(True))], margem=2.0)

# %% [markdown]
# Mesmo dobrando a PD de quem o modelo nunca viu, a política cumpre os limites. Negar esses casos custa aprovação e
# fura o volume no severo, sem melhorar o ROI.
#
# **Decisão 6: aceitar pela regra da faixa e avaliar a política final com a PD dos fora do perfil dobrada.**
#
# ## 7. Checagem final e apetite de risco

# %%
FINAL = politica.montar_regras(4, ESCADA, prazo_max=60, entrada_min=0.10, aceita_fora_perfil=True)
DELTA_AGRESSIVA = 0.003
AGRESSIVA = politica.montar_regras(
    4, {s: min(0.035, t + DELTA_AGRESSIVA) for s, t in ESCADA.items()}, prazo_max=60, entrada_min=0.10, aceita_fora_perfil=True
)
MARGEM = 2.0
paineis = {}
for chave, nome, regras in [
    ("final", "Política final (robusta no severo)", FINAL),
    ("agressiva", "Variante agressiva (+0,3 ponto)", AGRESSIVA),
]:
    p = politica.painel(modelo, base, cortes, regras, MARGEM)
    paineis[chave] = p
    print(nome)
    display(pd.DataFrame({
        "aprovação": p["aprovacao"].map("{:.1%}".format),
        "contratos": p["contratos"].round().astype(int),
        "volume (R$ mi)": (p["volume"] / 1e6).round(1),
        "inadimplência": p["inadimplencia"].map("{:.1%}".format),
        "prazo médio (anos)": p["prazo_medio_anos"].round(2),
        "ROI a.a.": p["roi"].map("{:.1%}".format),
        "limites ok": p["ok_limites"],
    }))

# %% [markdown]
# A variante agressiva sobe o ROI para ~18,6% no base, mas, se o cliente for tão sensível quanto no severo, fura a
# inadimplência e o volume. Cada limite furado corta a nota da política pela metade.
#
# **Decisão 7: política robusta.** ROI de 16,0% a 16,6% e todos os limites cumpridos nos três cenários, com o estresse.
#
# ### A tabela de regras

# %%
_, detalhe = politica.simular(modelo, base, cortes, FINAL, politica.CENARIOS[1], MARGEM)
w = detalhe["aceite"]
g = detalhe.assign(
    wj=w * detalhe["juros_esperados"], wp=w * detalhe["perda_esperada"], wv=w * detalhe["valor_financiado"],
    wz=w * detalhe["prazo_meses"], w=w, wpd=w * detalhe["pd_simulada"],
).groupby("score_1a10")[["wj", "wp", "wv", "wz", "w", "wpd"]].sum()
tabela_final = FINAL.copy()
tabela_final["propostas_C"] = base["score_1a10"].value_counts().reindex(tabela_final.index, fill_value=0)
tabela_final["aceite_base"] = (g["w"] / tabela_final["propostas_C"]).reindex(tabela_final.index)
tabela_final["pd_simulada_base"] = (g["wpd"] / g["w"]).reindex(tabela_final.index)
tabela_final["roi_base"] = ((g["wj"] - g["wp"]) / g["wv"] / (g["wz"] / g["w"] / 12)).reindex(tabela_final.index)
display(tabela_final.sort_index(ascending=False))

# %% [markdown]
# ## 8. Saídas

# %%
tabela_final.to_csv(SAIDAS / "tabela_politica.csv", encoding="utf-8")
submissao = politica.arquivo_submissao(base, FINAL)
politica.validar_coerencia(submissao, base, FINAL, cortes)
submissao.to_csv(SAIDAS / "politica_base_C.csv", index=False)
assert len(submissao) == 5000
print(submissao["decisao"].value_counts().to_dict())
display(submissao.head())

# Números para o resumo de segunda (scripts/gerar_resumo.py lê este arquivo).
resultados_politica = {
    "margem_fora_perfil": MARGEM,
    "delta_agressiva": DELTA_AGRESSIVA,
    "cenarios": [c.__dict__ for c in politica.CENARIOS],
    "final": paineis["final"].reset_index().to_dict("records"),
    "agressiva": paineis["agressiva"].reset_index().to_dict("records"),
    "regras": tabela_final.reset_index().to_dict("records"),
    "decisoes": submissao["decisao"].value_counts().to_dict(),
}
with open(SAIDAS / "resultados_politica.json", "w", encoding="utf-8") as arquivo:
    json.dump(resultados_politica, arquivo, ensure_ascii=False, indent=2)
