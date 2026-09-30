# Modelo de PD 90/12 — Desafio AutoCred (desenho)

- **Data:** 19/09/2026
- **Responsável:** papel de Modelagem (PD) do grupo
- **Status:** aprovado em conversa; aguardando revisão do documento

## 1. Objetivo

Construir um modelo de *application scoring* que estime a **PD 90/12**: a probabilidade de um contrato atingir 90 dias de atraso nos 12 meses seguintes à concessão. O objetivo é maximizar o **AUC na Base B** (out-of-time, jan–jun/2025). A PD e o score 1–10 também alimentam a política de crédito aplicada na Base C.

**Como a nota do modelo é composta (slide 11):**

| Critério | Pontos |
|---|---|
| AUC na Base B, relativo ao melhor grupo | 30 |
| Qualidade técnica: sem vazamento, split correto, Pipeline, reprodutibilidade | 10 |

O KS é reportado como métrica secundária. O dicionário cita "Gini", que é equivalente ao AUC (Gini = 2 × AUC − 1).

## 2. Escopo

**Dentro:**
- análise exploratória;
- pré-processamento em Pipeline;
- seleção de variáveis;
- torneio de modelos com validação temporal;
- escolha do campeão, calibração e retreino;
- escoragem das Bases B e C;
- faixas de score 1–10;
- perda esperada (PD × EAD × LGD) com as tabelas dadas;
- resumo de 1 página para a reunião do grupo (21/09).

**Fora:**
- a política de crédito (corte, taxa, prazo, entrada), que é do colega de política e pode ser exercitada depois pelo usuário;
- a modelagem de EAD e LGD, que vêm prontos;
- o documento de política e a defesa.

**Premissa:** o documento "AutoCred — Regras da Competição" não foi recebido. Seguimos o enunciado, os slides e os CSVs de exemplo em `entregaveis/`.

## 3. Dados

| Base | Conteúdo | Uso |
|---|---|---|
| A | 10.000 contratos, jan/2022–dez/2024, com alvo `default_90_12` (média de 8,26%) | treino e validação temporal |
| B | 3.000 contratos, jan–jun/2025, sem alvo | escoragem → submissão |
| C | 5.000 propostas, jul–dez/2025, sem taxa nem prazo contratado | escoragem → insumo da política |

Os arquivos ficam em `../bases/` e são lidos de lá, sem cópia para o repositório.

**Fatos que orientam o desenho**, levantados no diagnóstico inicial:

- **Vazamento.** `qtd_parcelas_em_atraso_12m` é apurada depois da concessão. Sozinha, tem AUC de 0,96 na Base A, e está zerada nas Bases B e C.
- **Mudança no tempo.** A inadimplência foi de 8,7% em 2022, 8,9% em 2023 e 7,2% em 2024.
- **Ausência informativa.** Com `score_bureau` ausente (3,3% dos casos), a inadimplência sobe para 11,9%. A renda está ausente em 7,7% dos casos e o tempo de emprego em 12,1%.
- **Viés de seleção.** A Base A só tem clientes com `score_bureau` ≥ 460 (ou ausente), `qtd_restricoes_ativas` ≤ 2 e LTV ≤ 0,95. Na Base C, 39,5% das propostas violam pelo menos um desses limites.
- **Sinal fraco.** A melhor variável isolada é `score_bureau`, com AUC de 0,61.

## 4. Variáveis

### 4.1 Proibidas (nunca entram no modelo)

- `qtd_parcelas_em_atraso_12m`, por vazamento.
- `default_90_12`, que é o alvo.
- `mes_default`, `ead_realizado`, `lgd_realizado` e `perda_financeira`, que são valores realizados.
- `id_contrato` e `id_proposta`, que são identificadores.
- `data_originacao`, usada apenas para separar treino e teste.
- `ano_modelo`. Ela é redundante com `idade_veiculo_anos` e muda com o calendário, o que prejudica a validação out-of-time.

Uma trava no código (`assert`) interrompe a execução se qualquer coluna proibida chegar ao Pipeline.

### 4.2 Candidatas

Todas estão disponíveis na concessão:

- **Numéricas:** `valor_bem`, `valor_entrada`, `valor_financiado`, `ltv`, `prazo_meses`, `parcela_mensal`, `comprometimento_renda`, `idade_veiculo_anos`, `idade_cliente`, `renda_mensal_declarada`, `tempo_emprego_meses`, `score_bureau`, `qtd_restricoes_ativas`, `qtd_consultas_bureau_3m`.
- **Categóricas:** `canal_originacao`, `ocupacao`, `tipo_residencia`, `possui_avalista`.
- **Apenas no conjunto completo:** `taxa_juros_am`.
- **Derivadas:** indicadores de ausência de `score_bureau`, `renda_mensal_declarada` e `tempo_emprego_meses`.

### 4.3 Dois conjuntos em disputa

- **Portátil:** todas as candidatas, exceto `taxa_juros_am`. Todas as variáveis do portátil existem na Base C ou podem ser recalculadas a partir da oferta (ver seção 9).
- **Completo:** o portátil mais `taxa_juros_am`, que reflete a precificação antiga.

**Regra de decisão:** o completo só é adotado se ganhar do portátil por pelo menos **0,005 de AUC médio** nas duas janelas (seção 5). Nesse caso, ele serve apenas para a submissão da Base B, e o portátil continua servindo à Base C. Caso contrário, um único modelo portátil serve às duas bases.

## 5. Validação

**Janelas temporais:**

| Janela | Treino | Teste |
|---|---|---|
| J1 | 2022 | 2023 |
| J2 | 2022–2023 | 2024 |

**Métricas:**
- A métrica de escolha é o **AUC médio de J1 e J2**.
- O **KS** é reportado nas duas janelas.
- O AUC de J2 vem com intervalo de confiança de 95% por bootstrap, com 1.000 reamostragens.
- As comparações entre modelos usam **bootstrap pareado** da diferença de AUC em J2. Esse resultado é informativo e não entra na regra de decisão.

**Princípio de complexidade.** Qualquer complexidade extra só entra se ganhar pelo menos **0,005 de AUC médio** sobre a alternativa mais simples. Vale para:
- boosting ou random forest contra a logística;
- ensemble contra o melhor modelo individual;
- conjunto completo contra o portátil.

Diferenças menores são tratadas como empate, e o empate favorece o mais simples.

A ordem de complexidade, da mais simples para a mais complexa, é:

1. logística;
2. boosting monotônico, que é mais restrito e mais fácil de defender;
3. random forest, XGBoost e LightGBM sem restrição;
4. ensemble.

O campeão começa como o melhor candidato do nível mais simples. Ele só é trocado por um candidato mais complexo que o supere em pelo menos 0,005.

**Isolamento.** Todo pré-processamento que aprende com o dado (imputação, codificação, padronização) fica dentro de um `sklearn.pipeline.Pipeline` e é ajustado apenas no treino de cada janela.

## 6. Modelos do torneio

| Modelo | Pré-processamento | Busca de hiperparâmetros |
|---|---|---|
| Regressão logística | mediana + indicador de ausência; one-hot; `log1p` em valores em reais e renda; padronização | `C` em grade log de 0,001 a 10 |
| Random forest | mediana + indicador; codificação ordinal das categorias | `min_samples_leaf`, `max_features`, `max_depth` |
| XGBoost | ausentes nativos + indicador; codificação ordinal | profundidade, taxa de aprendizado, número de árvores, `min_child_weight`, amostragem de linhas e colunas, regularização |
| LightGBM | ausentes nativos + indicador; codificação ordinal | `num_leaves`, taxa de aprendizado, número de árvores, `min_child_samples`, amostragem, regularização |
| Boosting monotônico | igual ao melhor boosting | mesma busca, com restrições monotônicas |

Nas árvores, as categorias usam codificação ordinal em vez de one-hot. Assim a ordem das colunas é conhecida antes do treino, o que as restrições monotônicas exigem. Com 2 a 5 categorias por variável, as árvores lidam bem com essa codificação.

**Detalhes da busca e do torneio:**
- A busca é aleatória, com 25 configurações por modelo e semente fixa.
- Cada configuração é avaliada pelo AUC médio de J1 e J2.
- Não há *early stopping* sobre a janela de teste.
- O **ensemble** é a média simples das probabilidades dos dois melhores modelos.

**Restrições monotônicas:**

| Direção | Variáveis |
|---|---|
| Risco aumenta com a variável | `qtd_restricoes_ativas`, `qtd_consultas_bureau_3m`, `ltv`, `comprometimento_renda` |
| Risco diminui com a variável | `score_bureau`, `renda_mensal_declarada`, `tempo_emprego_meses` |

Se alguma direção contrariar o que se vê nos dados de treino, ela é retirada e o motivo é registrado no notebook.

**Explicabilidade.** A importância das variáveis é medida por permutação na janela J2. Os efeitos das 5 principais aparecem em gráficos de dependência parcial.

**Contingência de ambiente.** Se XGBoost ou LightGBM não instalarem no Python 3.14, eles são substituídos por `HistGradientBoostingClassifier` do scikit-learn, que também aceita restrições monotônicas.

## 7. Calibração e modelo final

1. **Retreino.** O campeão, com a configuração vencedora, é retreinado na **Base A inteira** (2022–2024).
2. **Nível da PD.** Mantemos o nível natural do treino, perto de 8,3%, que é a média do ciclo 2022–24. É uma escolha conservadora e fica documentada. A comparação com a taxa de 2024 (7,2%) aparece no resumo.
3. **Forma da curva.** Avaliamos a curva de calibração em J2, comparando PD prevista com inadimplência observada por decil. A inclinação é o coeficiente de uma regressão logística do alvo de 2024 sobre `logit(PD prevista)`. Se ela ficar fora de 0,8–1,2, ou se o campeão for random forest, aplicamos uma calibração sigmoide (Platt).
   - A sigmoide é ajustada sobre as previsões fora do tempo de J1 e J2 empilhadas.
   - Depois, é aplicada à saída do modelo final.
   - Como é uma transformação monotônica, não altera o AUC.
   - Efeito colateral aceito: quando aplicada, a sigmoide também leva o nível médio da PD para a média de 2023–24 (cerca de 8%). O notebook registra esse efeito.
4. **Salvamento.** O modelo final é salvo com `joblib`.

## 8. Faixas de score 1–10

- **Cortes:** os decis da PD do modelo final na Base B, que é a população aprovada mais recente. O score 1 é o decil de maior PD e o score 10 o de menor. Propostas da Base C abaixo do menor corte ou acima do maior caem nas faixas das pontas.
- **Tabela por faixa:**
  - intervalo de PD;
  - quantidade de contratos;
  - PD média;
  - inadimplência observada no decil equivalente de J2 (2024), comparada por posição no ranking;
  - fator de EAD médio;
  - LGD média;
  - perda esperada em % do valor financiado.
- **Alternativa para discussão com a política:** faixas por PD fixa em escala geométrica. Ela aparece no notebook, mas não é o padrão.

## 9. EAD, LGD e perda esperada

- **Fórmula:** `PE = PD × (fator_EAD[prazo, faixa_LTV] × valor_financiado) × LGD[faixa_idade, faixa_LTV]`, subtraindo `0,061` da LGD quando `possui_avalista = "Sim"`.
- **Faixas de LTV, fechadas à esquerda:** [0; 0,60), [0,60; 0,70), [0,70; 0,80), [0,80; 0,90) e [0,90; ∞). Na conferência de 19/09, essa convenção reproduziu exatamente as tabelas a partir dos realizados da Base A. A convenção fechada à direita não reproduziu.
- **Faixas de idade do veículo:** 0–2, 3–5, 6–8 e 9 anos ou mais.
- **Conferência:** o fator de EAD médio realizado (`ead_realizado / valor_financiado`) e a LGD média realizada da Base A, por faixa, são comparados com as tabelas. Se aparecer divergência sistemática nas fronteiras, a premissa de fronteira é revista e a decisão, registrada.
- **Parcela:** Tabela Price, `PMT = PV × i / (1 − (1 + i)^−n)`. A fórmula é validada contra `parcela_mensal` da Base A.

### Escoragem da Base C

Nas condições pedidas pelo cliente:

- `valor_financiado` = `valor_financiado_desejado`
- `ltv` = `ltv_desejado`
- `valor_entrada` = `valor_entrada_desejada`
- `prazo_meses` = `prazo_desejado_meses`
- taxa de referência = média da Base A, **0,01589 a.m.**
- `parcela_mensal` = Price(valor financiado, taxa, prazo)
- `comprometimento_renda` = parcela ÷ renda. Fica ausente quando a renda está ausente e é imputado pelo Pipeline.

Cada proposta recebe também o indicador `fora_perfil_historico`, verdadeiro quando `score_bureau` < 460, ou `qtd_restricoes_ativas` > 2, ou `ltv_desejado` > 0,95.

**Função `prever_pd(propostas, taxa_am, prazo_meses, pct_entrada)`.** Recalcula LTV, valor financiado, parcela e comprometimento para outra oferta e devolve a PD, o score, o EAD, a LGD e a PE.

## 10. Saídas (`modelo_pd/saidas/`)

| Arquivo | Conteúdo |
|---|---|
| `submissao_modelo_base_B.csv` | `id_contrato,pd`: 3.000 linhas, PD com 6 casas decimais |
| `base_C_escorada.xlsx` | aba `propostas`: id, PD, score 1–10, fator EAD, EAD, LGD, PE (R$ e %), `fora_perfil_historico`; aba `faixas`: tabela da seção 8; aba `leia_me`: premissas e alertas |
| `faixas_score.csv` | tabela da seção 8 |
| `modelo_pd.joblib` | Pipeline final (e o modelo portátil, se diferente) |
| `resumo_segunda.html` | resumo de 1 página: AUC e KS por modelo, campeão, variáveis principais, armadilhas evitadas, faixas, limitações |

**Alertas escritos para o colega de política,** que vão na aba `leia_me` e no resumo:

1. O modelo não enxerga o efeito de seleção adversa da taxa descrito na seção 8 do enunciado.
2. A PD das propostas fora do perfil histórico é uma extrapolação e merece margem de segurança.
3. Na Base C, o comprometimento de renda é recalculado com a taxa de referência. Na Base A, ele nunca está ausente.

## 11. Estrutura do projeto

```
modelo_pd/
  README.md
  requirements.txt          versões fixas
  .gitignore                .venv, caches, saidas/ (dados são da Analítica)
  docs/specs/               este documento
  docs/plans/               plano de execução
  src/autocred/
    dados.py                leitura das bases, listas de colunas, colunas proibidas
    features.py             variáveis derivadas, recálculo da oferta (Base C)
    modelos.py              Pipelines e espaços de busca
    avaliacao.py            AUC, KS, bootstrap, janelas temporais
    perda.py                Price, fator EAD, LGD, perda esperada
    faixas.py               cortes 1–10 e tabela por faixa
    escoragem.py            prever_pd e escoragem de B e C
  tests/                    pytest das funções de src/
  notebooks/
    01_diagnostico.ipynb
    02_torneio_modelos.ipynb
    03_modelo_final.ipynb
  saidas/
```

- O ambiente é um `.venv` local com Python 3.14.
- As bibliotecas são pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, openpyxl, joblib, jupyter, jupytext e pytest. Todas têm versão compatível com o Python 3.14, conferido em 19/09.
- Cada notebook tem uma fonte `.py` em formato *percent*, que é a versão guardada no git. O `.ipynb` com as saídas é gerado a partir dela com jupytext e executado com nbconvert.
- Os notebooks trazem explicações didáticas em português. O cálculo fica em `src/`, para ser reaproveitado pelo colega e depois no Databricks.

## 12. Checagens

- **Testes (pytest):**
  - KS contra um exemplo calculado à mão;
  - fator de EAD e LGD nas fronteiras de LTV (0,60 / 0,70 / 0,80 / 0,90) e de idade (2/3, 5/6, 8/9);
  - ajuste de avalista;
  - Price contra `parcela_mensal` da Base A (tolerância de R$ 0,05);
  - faixas cobrindo 1–10 sem buracos;
  - `prever_pd` sem alterar a oferta reproduzindo a escoragem padrão.
- **Travas nos notebooks:**
  - nenhuma coluna proibida nas variáveis;
  - submissão com exatamente 3.000 linhas;
  - IDs iguais e na mesma ordem da Base B;
  - PD em (0, 1) e nenhum valor vazio.
- **Reprodutibilidade:** semente global fixa. Os três notebooks são executados do zero duas vezes, e o AUC tem de ser idêntico.

## 13. Riscos e limitações

| Risco | Mitigação |
|---|---|
| AUC de J2 com ruído de cerca de ±0,02, com só ~240 inadimplentes em 2024 | média de duas janelas, bootstrap e princípio de complexidade |
| Superajuste à validação na busca de hiperparâmetros | busca pequena (25 configurações) e monotonicidade como variante |
| Base C fora do perfil de treino (reject inference) | indicador `fora_perfil_historico`, alerta escrito e monotonicidade |
| Mudança no nível de inadimplência (2024 menor) | nível conservador documentado; o AUC não é afetado |
| Bibliotecas indisponíveis no Python 3.14 | contingência com `HistGradientBoostingClassifier` |

## 14. Cronograma

| Quando | Etapa |
|---|---|
| Sáb 19/09 | spec, plano, ambiente, `01_diagnostico` |
| Dom 20/09 | `02_torneio_modelos`, `03_modelo_final`, saídas |
| Seg 21/09, antes da reunião | `resumo_segunda.html` |
| 22–25/09 | ajustes com o grupo; submissão até sex 25/09, 23h59 |
| Sáb 26/09 | apuração e defesa |
