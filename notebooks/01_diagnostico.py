# %% [markdown]
# # 01 · Diagnóstico dos dados
#
# **Objetivo:** conhecer as bases antes de modelar e achar as armadilhas que derrubariam o modelo na Base B.
#
# 1. Inadimplência por safra
# 2. Poder de cada variável sozinha (AUC univariado), incluindo a variável que "vaza" o futuro
# 3. Valores ausentes que carregam risco
# 4. Categorias
# 5. Viés de seleção: a Base C é diferente da A
# 6. Conferência dos parâmetros de EAD e LGD

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autocred import dados, perda
from autocred.avaliacao import auc

pd.set_option("display.float_format", "{:.4f}".format)
A = dados.carregar_base("A")
B = dados.carregar_base("B")
C = dados.carregar_base("C")
A["ano"] = dados.ano(A)
print("Base A:", A.shape, "| Base B:", B.shape, "| Base C:", C.shape)

# %% [markdown]
# ## 1. Inadimplência por safra
#
# A **PD 90/12 observada** é a média do alvo `default_90_12`: a fração de contratos que chegaram a 90 dias
# de atraso nos 12 meses após a concessão. Se ela muda de um ano para outro, o passado não é igual ao futuro.
# Por isso validamos **por tempo** (treina no passado, testa no ano seguinte) e nunca com split aleatório.

# %%
por_ano = A.groupby("ano")[dados.ALVO].agg(contratos="size", inadimplencia="mean")
display(por_ano)
ax = por_ano["inadimplencia"].plot.bar(color="#534AB7", figsize=(6, 3), title="Inadimplência 90/12 por safra", rot=0)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
plt.show()

# %% [markdown]
# ## 2. Quanto cada variável ordena o risco sozinha
#
# O **AUC** mede a ordenação: é a chance de um inadimplente sorteado ter valor "mais arriscado" que um adimplente
# sorteado. 0,5 = moeda; 1,0 = perfeito. Aqui cada variável é usada sozinha, sem modelo.

# %%
candidatas = dados.NUMERICAS + ["taxa_juros_am", "qtd_parcelas_em_atraso_12m"]
y = A[dados.ALVO].to_numpy()
linhas = []
for coluna in candidatas:
    presente = A[coluna].notna().to_numpy()
    a = auc(y[presente], A.loc[presente, coluna])
    linhas.append({"variavel": coluna, "auc": max(a, 1 - a), "direcao": "sobe o risco" if a >= 0.5 else "desce o risco"})
univariado = pd.DataFrame(linhas).sort_values("auc", ascending=False).reset_index(drop=True)
display(univariado)

# %% [markdown]
# ### A variável que vaza o futuro
#
# `qtd_parcelas_em_atraso_12m` tem AUC perto de **0,96** sozinha. Parece mágica, mas é apurada **depois** da
# concessão: conta as parcelas atrasadas nos 12 meses seguintes. Na hora de aprovar o crédito ela não existe.
# Nas Bases B e C ela está **zerada**. Um modelo que dependesse dela veria todo mundo igual e erraria a ordenação.
# Por isso ela está na lista `dados.PROIBIDAS`, e o Pipeline se recusa a usá-la.

# %%
display(pd.DataFrame({
    "Base A": A["qtd_parcelas_em_atraso_12m"].describe(),
    "Base B": B["qtd_parcelas_em_atraso_12m"].describe(),
    "Base C": C["qtd_parcelas_em_atraso_12m"].describe(),
}))
assert (B["qtd_parcelas_em_atraso_12m"] == 0).all() and (C["qtd_parcelas_em_atraso_12m"] == 0).all()

# %% [markdown]
# Fora a variável proibida, a melhor ordenação individual é a do `score_bureau` (≈ 0,61). O sinal de cada
# variável é fraco: o ganho vai vir da **combinação**, que é o trabalho do modelo.
#
# ## 3. Ausentes que carregam risco
#
# Um valor em branco não é só "falta de dado". Quem não tem histórico no bureau, por exemplo, pode ser mais
# arriscado. Por isso criamos um indicador `ausente_<coluna>` antes de preencher o valor.

# %%
ausentes = pd.DataFrame({
    "pct_ausente": A[dados.COM_AUSENTES].isna().mean(),
    "inad_quando_ausente": [A.loc[A[c].isna(), dados.ALVO].mean() for c in dados.COM_AUSENTES],
    "inad_quando_presente": [A.loc[A[c].notna(), dados.ALVO].mean() for c in dados.COM_AUSENTES],
}, index=dados.COM_AUSENTES)
display(ausentes)

# %% [markdown]
# ## 4. Categorias

# %%
for coluna in dados.CATEGORICAS:
    display(A.groupby(coluna)[dados.ALVO].agg(contratos="size", inadimplencia="mean").sort_values("inadimplencia"))

# %% [markdown]
# ## 5. Viés de seleção: a Base C não é igual à A
#
# As Bases A e B só têm clientes **aprovados** pela política antiga. A Base C tem todo mundo que pediu crédito,
# inclusive perfis que eram recusados. O modelo nunca viu esses perfis, então a PD deles é uma **extrapolação**.
# Este é o problema clássico da **inferência de rejeitados**.

# %%
comparar = ["score_bureau", "qtd_restricoes_ativas", "qtd_consultas_bureau_3m", "renda_mensal_declarada", "idade_veiculo_anos"]
display(pd.DataFrame({"Base A": A[comparar].mean(), "Base B": B[comparar].mean(), "Base C": C[comparar].mean()}))
display(pd.Series({
    "score_bureau mínimo na Base A": A["score_bureau"].min(),
    "restrições máximas na Base A": A["qtd_restricoes_ativas"].max(),
    "LTV máximo na Base A": A["ltv"].max(),
}))
fora = (C["score_bureau"] < 460) | (C["qtd_restricoes_ativas"] > 2) | (C["ltv_desejado"] > 0.95)
print(f"Propostas da Base C fora do perfil histórico: {fora.mean():.1%}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(A["score_bureau"].dropna(), bins=40, alpha=0.6, density=True, label="Base A (aprovados)", color="#534AB7")
ax.hist(C["score_bureau"].dropna(), bins=40, alpha=0.6, density=True, label="Base C (propostas)", color="#1D9E75")
ax.axvline(460, color="#444441", linestyle="--", linewidth=1)
ax.set_title("Score de bureau: aprovados x propostas")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. Conferência dos parâmetros de EAD e LGD
#
# O enunciado diz que as tabelas saem dos valores realizados da Base A. Conferimos, e descobrimos a convenção
# das faixas de LTV: **fechadas à esquerda**. Um LTV de exatamente 60% cai em "60% a 70%".

# %%
tab = perda.carregar_tabelas()
d = A[A[dados.ALVO] == 1].copy()
d["f_ltv"] = perda.faixa_ltv(d["ltv"])
d["f_idade"] = perda.faixa_idade(d["idade_veiculo_anos"])
fator_real = (d["ead_realizado"] / d["valor_financiado"]).groupby([d["prazo_meses"], d["f_ltv"]]).mean().unstack()
lgd_real = d["lgd_realizado"].groupby([d["f_idade"], d["f_ltv"]]).mean().unstack()
dif_ead = np.abs(fator_real.to_numpy() - tab.fator_ead).max()
dif_lgd = np.abs(lgd_real.to_numpy() - tab.lgd).max()
print(f"Maior diferença entre realizado e tabela | EAD: {dif_ead:.4f} | LGD: {dif_lgd:.4f}")

# %% [markdown]
# ## Conclusões que viram regra no modelo
#
# | Achado | Decisão |
# |---|---|
# | `qtd_parcelas_em_atraso_12m` vaza o futuro | proibida (trava no código) |
# | Inadimplência muda por safra | validação out-of-time: J1 (2022→2023) e J2 (2022–23→2024) |
# | Ausência de bureau/renda/emprego carrega risco | indicadores `ausente_*` dentro do Pipeline |
# | Sinal individual fraco (melhor AUC ≈ 0,61) | torneio de modelos que combinam variáveis |
# | ~40% da Base C fora do perfil histórico | alerta para a política e indicador `fora_perfil_historico` |
# | Tabelas de EAD/LGD conferem com a Base A | usamos as tabelas com faixas fechadas à esquerda |
