# %% [markdown]
# # 03 · Modelo final, escoragem e faixas de score
#
# 1. Retreinar o campeão com a Base A inteira (2022–24) e decidir a calibração
# 2. Escorar a Base B e gravar a submissão
# 3. Montar as faixas 1–10 e a perda esperada por faixa
# 4. Escorar a Base C para a política
# 5. Explicar o modelo: variáveis que mais pesam e como a PD reage a elas
# 6. Gravar as saídas

# %%
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from autocred import dados, escoragem, faixas, modelos
from autocred.avaliacao import dependencia_parcial, mascaras_janela

SAIDAS = dados.PASTA_SAIDAS
with open(SAIDAS / "campeao.json", encoding="utf-8") as arquivo:
    campeao = json.load(arquivo)
descricao = campeao["descricao"]
conjunto_b = campeao["conjunto_b"]

A = dados.carregar_base("A")
B = dados.carregar_base("B")
C = dados.carregar_base("C")
anos = dados.ano(A)
y = A[dados.ALVO].to_numpy()
y_2024 = y[anos == 2024]
print(f"Campeão: {campeao['candidato']} | Base B usa o conjunto: {conjunto_b}")

# %% [markdown]
# ## 1. Retreino e calibração
#
# Primeiro medimos mais uma vez fora do tempo (J1 e J2). Depois treinamos com **toda** a Base A: agora que a
# nota foi medida, não há por que esconder 2024, o ano mais parecido com 2025.
#
# **Calibração:** o AUC só mede ordenação, mas a política precisa da PD no nível certo. A curva abaixo compara,
# em cada decil de 2024, a PD prevista com a inadimplência que de fato aconteceu. A inclinação ideal é 1.

# %%
modelo_portatil, info_portatil, prev_portatil = modelos.treinar_final(descricao, A, y, anos, "portatil")
if conjunto_b == "completo":
    modelo_b, info_b, _ = modelos.treinar_final(descricao, A, y, anos, "completo")
else:
    modelo_b, info_b = modelo_portatil, info_portatil
print({k: round(v, 4) if isinstance(v, float) else v for k, v in info_portatil.items()})

p_j2 = prev_portatil["J2"]
decil = faixas.atribuir_score(p_j2, faixas.cortes_decis(p_j2))
curva = pd.DataFrame({"prevista": p_j2, "observada": y_2024}).groupby(decil).mean()
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot(curva["prevista"], curva["observada"], "o-", color="#534AB7", label="decis de 2024")
limite = float(curva.max().max()) * 1.1
ax.plot([0, limite], [0, limite], "--", color="#888780", label="calibração perfeita")
ax.set_xlabel("PD prevista")
ax.set_ylabel("inadimplência observada")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. Submissão da Base B

# %%
p_b_submissao = escoragem.prever(modelo_b, B, conjunto_b)
submissao = escoragem.salvar_submissao_b(B["id_contrato"], p_b_submissao, B, SAIDAS / "submissao_modelo_base_B.csv")
display(submissao.head())
print(f"PD média na Base B: {submissao['pd'].mean():.4f}")

# %% [markdown]
# ## 3. Faixas 1–10 e perda esperada
#
# Os cortes são os **decis da PD na Base B**, a carteira aprovada mais recente: cada faixa fica com 10% dela.
# Score 1 = maior risco, 10 = menor. A coluna `inadimplencia_observada_J2` mostra o que aconteceu de verdade em
# 2024 no decil equivalente (modelo treinado só até 2023). É a prova de que a ordenação se sustenta.
#
# **Perda esperada (% do valor financiado) = PD × fator de EAD × LGD.** É o piso que a taxa da faixa precisa cobrir.

# %%
p_b_portatil = escoragem.prever(modelo_portatil, B)
cortes = faixas.cortes_decis(p_b_portatil)
escorada_b = escoragem.escorar(modelo_portatil, B, cortes)
tabela = faixas.tabela_faixas(escorada_b)
tabela["inadimplencia_observada_J2"] = faixas.inadimplencia_por_decil(y_2024, p_j2)
resultado_c = escoragem.prever_pd(modelo_portatil, C, cortes)
tabela["qtd_base_C"] = resultado_c["score_1a10"].value_counts().reindex(range(1, 11), fill_value=0)
tabela["pct_fora_perfil_C"] = resultado_c.groupby("score_1a10")["fora_perfil_historico"].mean().reindex(range(1, 11))
display(tabela)

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(tabela.index, tabela["perda_esperada_pct_media"], color="#534AB7")
ax.set_xticks(range(1, 11))
ax.set_xlabel("score (1 = pior)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
ax.set_title("Perda esperada média por faixa (% do valor financiado)")
plt.show()

# %% [markdown]
# ### Alternativa para discutir com a política: faixas por PD em escala geométrica
#
# Em vez de 10% da carteira por faixa, cada faixa multiplica a PD pelo mesmo fator. As pontas ficam mais
# "puras" (menos gente, risco mais homogêneo) e o meio fica mais cheio.

# %%
cortes_geo = faixas.cortes_geometricos(p_b_portatil.min(), p_b_portatil.max())
alternativa = pd.DataFrame({
    "qtd_B_decis": pd.Series(faixas.atribuir_score(p_b_portatil, cortes)).value_counts(),
    "qtd_B_geometrica": pd.Series(faixas.atribuir_score(p_b_portatil, cortes_geo)).value_counts(),
    "qtd_C_geometrica": pd.Series(faixas.atribuir_score(resultado_c["pd"], cortes_geo)).value_counts(),
}).reindex(range(1, 11)).fillna(0).astype(int)
display(alternativa)

# %% [markdown]
# ## 4. Base C: o que muda no mar aberto
#
# A PD da Base C foi calculada nas condições que o cliente pediu (prazo e entrada desejados, com taxa de
# referência de 1,589% a.m.). Repare quanta gente cai nas faixas 1–3, e quantos estão **fora do perfil
# histórico**: para esses, a PD é extrapolação e merece margem de segurança.

# %%
print(f"PD média | Base B: {p_b_portatil.mean():.4f} | Base C: {resultado_c['pd'].mean():.4f}")
print(f"Base C fora do perfil histórico: {resultado_c['fora_perfil_historico'].mean():.1%}")
display(resultado_c.head())

# %% [markdown]
# ## 5. Explicando o modelo
#
# **Importância por permutação:** embaralhamos uma variável de cada vez em 2024 e medimos quanto o AUC cai.
# Quanto maior a queda, mais o modelo depende dela. Usamos o modelo treinado até 2023 para medir em dado
# que ele não viu.

# %%
X_portatil = A[dados.colunas_entrada("portatil")]
treino_j2, teste_j2 = mascaras_janela(anos, "J2")
modelo_j2 = modelos.construir_modelo(descricao, "portatil").fit(X_portatil.loc[treino_j2], y[treino_j2])
perm = permutation_importance(
    modelo_j2, X_portatil.loc[teste_j2], y_2024, scoring="roc_auc", n_repeats=10, random_state=modelos.SEMENTE
)
importancias = (
    pd.DataFrame({"variavel": X_portatil.columns, "queda_auc": perm.importances_mean, "desvio": perm.importances_std})
    .sort_values("queda_auc", ascending=False)
    .reset_index(drop=True)
)
display(importancias.head(10))

top = importancias.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(top["variavel"], top["queda_auc"], xerr=top["desvio"], color="#534AB7")
ax.set_title("Queda de AUC ao embaralhar a variável (2024)")
plt.show()

# %% [markdown]
# **Dependência parcial:** como a PD média da carteira muda quando mexemos só em uma variável. É a resposta
# para "o que acontece com o risco se o LTV sobe?".

# %%
amostra = X_portatil.sample(2000, random_state=modelos.SEMENTE)
principais = importancias["variavel"].head(5).tolist()
fig, eixos = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
for eixo, coluna in zip(eixos, principais):
    curva_dp = dependencia_parcial(modelo_portatil, amostra, coluna)
    if pd.api.types.is_numeric_dtype(amostra[coluna]):
        eixo.plot(curva_dp["valor"], curva_dp["pd_media"], color="#534AB7")
    else:
        eixo.bar(curva_dp["valor"].astype(str), curva_dp["pd_media"], color="#534AB7")
        eixo.tick_params(axis="x", rotation=45)
    eixo.set_title(coluna, fontsize=10)
eixos[0].set_ylabel("PD média")
plt.show()

# %% [markdown]
# ## 6. Gravando as saídas

# %%
tabela.to_csv(SAIDAS / "faixas_score.csv", encoding="utf-8")

leia_me = pd.DataFrame(
    [
        ("pd", "PD 90/12 estimada nas condições pedidas pelo cliente (prazo e entrada desejados), taxa de referência 1,589% a.m."),
        ("score_1a10", "1 = maior risco, 10 = menor. Cortes = decis da PD na Base B (carteira aprovada mais recente)."),
        ("fator_ead / ead", "Tabela do desafio por prazo e faixa de LTV (faixas fechadas à esquerda: 60% cai em 60–70%)."),
        ("lgd", "Tabela do desafio por idade do veículo e faixa de LTV; −0,061 quando há avalista."),
        ("perda_esperada", "PD × EAD × LGD, em R$; perda_esperada_pct = em % do valor financiado."),
        ("fora_perfil_historico", "Score de bureau < 460, mais de 2 restrições ou LTV > 95%: perfil que o modelo nunca viu."),
        ("Alerta 1", "O modelo não enxerga a seleção adversa da taxa: taxa alta tende a elevar a PD real além da prevista."),
        ("Alerta 2", "Para propostas fora do perfil histórico, a PD é extrapolação; use margem de segurança."),
        ("Alerta 3", "Na Base C o comprometimento de renda foi recalculado com a taxa de referência; sem renda, foi imputado."),
        ("Recalcular", "escoragem.prever_pd(modelo, propostas, cortes, taxa_am, prazo_meses, pct_entrada_minima) com saidas/modelo_pd.joblib."),
    ],
    columns=["item", "descricao"],
)
with pd.ExcelWriter(SAIDAS / "base_C_escorada.xlsx", engine="openpyxl") as planilha:
    resultado_c.to_excel(planilha, sheet_name="propostas", index=False)
    tabela.reset_index().to_excel(planilha, sheet_name="faixas", index=False)
    leia_me.to_excel(planilha, sheet_name="leia_me", index=False)

joblib.dump(
    {"modelo_portatil": modelo_portatil, "modelo_b": modelo_b, "conjunto_b": conjunto_b, "cortes": cortes, "descricao": descricao},
    SAIDAS / "modelo_pd.joblib",
)

resultados = {
    "campeao": campeao,
    "info_portatil": info_portatil,
    "info_b": info_b,
    "cortes": cortes.tolist(),
    "faixas": tabela.reset_index().to_dict("records"),
    "importancias": importancias.head(10).to_dict("records"),
    "pd_media_A": float(escoragem.prever(modelo_portatil, A).mean()),
    "pd_media_B": float(p_b_portatil.mean()),
    "pd_media_C": float(resultado_c["pd"].mean()),
    "inadimplencia_2024": float(y_2024.mean()),
    "pct_fora_perfil_C": float(resultado_c["fora_perfil_historico"].mean()),
}
with open(SAIDAS / "resultados_modelo.json", "w", encoding="utf-8") as arquivo:
    json.dump(resultados, arquivo, ensure_ascii=False, indent=2)

assert len(submissao) == 3000 and len(resultado_c) == 5000
assert resultado_c["score_1a10"].between(1, 10).all()
print("Saídas gravadas em saidas/")
