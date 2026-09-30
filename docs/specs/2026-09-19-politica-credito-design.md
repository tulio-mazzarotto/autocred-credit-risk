# Política de crédito na Base C — exercício guiado (desenho)

- **Data:** 19/09/2026
- **Contexto:** o usuário cuida do modelo de PD. Este exercício é para aprender a parte de política e levar uma proposta para comparar com a do colega responsável na reunião de segunda (21/09). **Não é a submissão oficial.**
- **Abordagem aprovada:** simulador com cenários de premissas de reação do cliente. O usuário decide cada alavanca, uma de cada vez.

## 1. Objetivo

Montar uma tabela de regras por score (1 a 10), com aprovar ou negar, taxa, prazo máximo, entrada mínima e tratamento para quem está fora do perfil histórico. A tabela deve:

- **cumprir os limites do conselho mesmo no cenário severo:** aprovação ≥ 35% das 5.000 propostas; inadimplência ≤ 8% dos contratos; volume contratado ≥ R$ 40 milhões; taxa ≤ 3,5% a.m.;
- **buscar o melhor ROI anual no cenário base:** o conselho pede mais de 15%.

## 2. Por que cenários

A Base C reage às ofertas, e o enunciado só declara a direção dos efeitos, não a intensidade:

- taxa alta derruba o aceite e aumenta a PD (seleção adversa);
- entrada maior reduz PD e LGD, mas derruba o aceite;
- prazo menor derruba o aceite.

Como há uma única submissão, a política precisa ser robusta a premissas que não conhecemos. Otimizar premissas inventadas daria uma falsa precisão.

## 3. Motor (`src/autocred/politica.py`)

**Regras.** A política é um DataFrame com índice de score 1–10 e as colunas `aprovar`, `taxa_am`, `prazo_max`, `entrada_min` e `aceita_fora_perfil`.

**Score da proposta.** É o score nas condições pedidas pelo cliente, com a taxa de referência de 1,589% a.m., usando os mesmos cortes do modelo. A oferta não muda a faixa da proposta.

**Oferta por proposta:**

- **Decisão:** APROVAR se a faixa aprova **e** (a proposta está dentro do perfil histórico **ou** a faixa aceita fora do perfil).
- **Taxa:** `min(taxa_am da faixa, 3,5%)`.
- **Prazo:** `min(prazo desejado, prazo_max da faixa)`.
- **Entrada:** `max(entrada desejada, entrada_min da faixa)`. O LTV e o valor financiado são recalculados.

O perfil histórico é o da política antiga: score de bureau ≥ 460 (ou ausente), no máximo 2 restrições e LTV desejado ≤ 95%.

**PD simulada.** Parte da PD do modelo nas condições ofertadas, via `features.oferta_no_formato_a` + `escoragem.escorar`, e aplica:

- **seleção adversa:** × `(1 + seleção_adversa) ^ Δtaxa`, onde Δtaxa = max(0, taxa − 1,589%) em pontos percentuais ao mês;
- **margem para quem está fora do perfil:** × `margem_fora_perfil`. É uma decisão do usuário, e o padrão é 1.

O resultado é limitado a 0,99.

**Aceite.** `aceite_base × fator_taxa ^ Δtaxa × (1 − queda_entrada × Δentrada/10 p.p.) × (1 − queda_prazo × meses cortados/12)`, com piso em 0. Taxa abaixo da referência não aumenta o aceite, uma premissa conservadora.

**Juros (Tabela Price):**

- Juros pagos nas primeiras `m` parcelas: `m × PMT − (PV − saldo_m)`.
- Sem default, a proposta paga todas as `n` parcelas.
- Com default, paga as parcelas até o mês do calote. O mês segue a aba `Distribuicao_Mes_Default` do arquivo de parâmetros.
- Juros esperados: `(1 − PD) × J(n) + PD × Σ P(m) × J(m)`.

**Perda esperada.** `PD × EAD × LGD`, pelas tabelas do desafio.

**Carteira.** Tudo é ponderado pelo aceite `w` de cada proposta aprovada:

- aprovação = aprovados ÷ 5.000;
- contratos = Σw;
- volume = Σw·PV;
- inadimplência = Σw·PD ÷ Σw;
- prazo médio (anos) = Σw·prazo ÷ Σw ÷ 12;
- ROI anual = [(Σw·juros − Σw·perda) ÷ volume] ÷ prazo médio em anos.

## 4. Cenários (ponto de partida; o usuário valida na decisão 1)

| Premissa | Brando | Base | Severo |
|---|---|---|---|
| `aceite_base` | 0,90 | 0,85 | 0,80 |
| `fator_taxa` (por +1 p.p. a.m.) | 0,86 | 0,74 | 0,61 |
| `queda_entrada` (por +10 p.p.) | 0,05 | 0,10 | 0,15 |
| `queda_prazo` (por 12 meses cortados) | 0,05 | 0,10 | 0,15 |
| `selecao_adversa` (por +1 p.p. a.m.) | 0,10 | 0,22 | 0,42 |

## 5. Roteiro guiado

Em cada decisão, o motor mostra aprovação, volume, inadimplência e ROI nos três cenários, e o usuário escolhe:

1. premissas;
2. corte;
3. taxa por faixa;
4. prazo máximo por faixa;
5. entrada mínima por faixa;
6. fora do perfil;
7. checagem final.

## 6. Entregas

- `notebooks/04_politica.py`: premissas, decisões com o porquê, tabela final e resultado nos três cenários.
- `saidas/tabela_politica.csv`: a regra por faixa.
- `saidas/politica_base_C.csv`: no formato do exemplo (`id_proposta, pd, score_1a10, decisao, taxa_am, prazo_meses, pct_entrada_minima`). As colunas seguem estas regras:
  - `pd` é a PD de referência (a que define o score);
  - `prazo_meses` é o prazo ofertado;
  - linhas NEGAR têm taxa, prazo e entrada vazios;
  - uma trava de coerência confere cada linha contra a tabela de regras.

## 7. Limitações declaradas

- As premissas de reação são inventadas e calibradas por bom senso. A robustez vem de cumprir os limites no cenário severo.
- A PD de quem está fora do perfil é extrapolação do modelo.
- O ROI não inclui custo de captação nem despesas, porque segue a definição do enunciado.
