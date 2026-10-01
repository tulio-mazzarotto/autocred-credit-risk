# AutoCred: modelo de PD e política de crédito

[![testes](https://github.com/tulio-mazzarotto/autocred-credit-risk/actions/workflows/testes.yml/badge.svg)](https://github.com/tulio-mazzarotto/autocred-credit-risk/actions/workflows/testes.yml)

*[Read in English](README.en.md)*

Projeto vencedor do desafio AutoCred, da mentoria JUMP (Analítica Educação), em setembro de 2026.
Ele tem duas partes: um modelo que estima a probabilidade de inadimplência (PD) de quem pede um
financiamento de veículo e uma política de crédito construída sobre esse modelo, que decide quem
aprovar, a que taxa, em que prazo e com quanta entrada.

**[Página interativa com a história do projeto e um simulador da política](https://tulio-mazzarotto.github.io/autocred-credit-risk/)**

## Resultado

O modelo foi medido pelo AUC numa base de teste de um período posterior ao do treino, e a política,
pelo ROI anual numa base de propostas novas. As duas medições usaram o desfecho real, que não
tínhamos.

| Indicador | Resultado | Exigência do conselho |
|---|---:|---:|
| AUC do modelo (Gini) | **0,7711** (0,5422) | |
| ROI anual | **16,53%** (projetado: 16,40%) | acima de 15% |
| Inadimplência da carteira | **4,75%** | no máximo 8% |
| Aprovação | **44,8%** | pelo menos 35% |
| Volume contratado | **R$ 50,1 mi** | pelo menos R$ 40 mi |
| Taxa média | **2,51% ao mês** (63,9% dos aprovados contrataram) | no máximo 3,5% ao mês |

Fonte: apuração da mentoria em 26/09/2026, transcrita em [`docs/resultados_oficiais.json`](docs/resultados_oficiais.json).

## O problema

A AutoCred é uma financeira fictícia de veículos que emprestava barato para uma clientela arriscada.
Dos 10.000 contratos do histórico (2022 a 2024), 8,3% chegaram a 90 dias de atraso no primeiro ano.
O conselho pediu uma política nova com ROI acima de 15% ao ano, respeitando quatro limites:

- aprovar pelo menos 35% das propostas;
- inadimplência de no máximo 8% da carteira;
- pelo menos R$ 40 milhões contratados;
- taxa de no máximo 3,5% ao mês.

As bases do desafio:

| Base | Linhas | Uso |
|---|---:|---|
| A | 10.000 contratos | desenvolvimento do modelo, com o desfecho |
| B | 3.000 contratos | teste do modelo (AUC) |
| C | 5.000 propostas | teste da política (ROI) |

O caminho do risco até o preço é a cadeia de perda esperada: **PD × EAD × LGD**. A PD é a chance de
não pagar; a EAD, quanto o cliente deve no momento do calote; a LGD, quanto se perde depois de retomar
e vender o carro. A taxa precisa cobrir essa perda e ainda deixar retorno.

## Decisões que fizeram a diferença

1. **Trava contra vazamento de dados.** A coluna `qtd_parcelas_em_atraso_12m` é apurada depois da
   concessão do crédito. Sozinha, ela dá AUC de 0,96 no treino, e vem zerada nas bases de teste.
   `dados.validar_sem_proibidas` impede que ela, o alvo ou qualquer valor realizado entre no modelo.
2. **Validação fora do tempo.** Cada candidato foi treinado em 2022 e testado em 2023, depois treinado
   em 2022 e 2023 e testado em 2024. Um modelo mais complexo só substituía um mais simples se ganhasse
   pelo menos 0,005 de AUC médio. A distribuição dos scores entre as safras de 2023 e 2024 também foi
   medida pelo PSI (Population Stability Index): 0,0015, muito abaixo do limite de atenção de 0,10.
3. **Restrições monotônicas.** O campeão, um LightGBM, é proibido de aprender relações sem sentido de
   negócio, como mais restrições no bureau reduzindo o risco. AUC médio fora do tempo de 0,737, contra
   0,667 da regressão logística.
4. **Aceite e seleção adversa.** Nem todo aprovado contrata, e taxas altas atraem justamente os clientes
   mais arriscados. O simulador (`politica.py`) projeta três cenários de reação do cliente, e a política
   escolhida cumpre os limites até no mais severo. Propostas fora do perfil histórico têm a PD dobrada
   por prudência.
5. **Preço ancorado no mercado.** As taxas vão de 2,3% a 2,9% ao mês conforme o score, comparadas com
   dados públicos do Banco Central sobre 45 instituições que financiam veículos
   ([`docs/benchmark/`](docs/benchmark/)).

## Camada Gold (desenho de produção)

Como o modelo e a política iriam para um Data Lake: arquitetura em camadas (Bronze, Silver e Gold),
quatro tabelas Gold (propostas escoradas, regras de preço, decisões de crédito e monitoramento do
score), consultas SQL para o motor de decisão em lote e para os painéis, e um protótipo que monta as
tabelas com o modelo e a política reais. O protótipo reproduz a decisão da submissão oficial nas 5.000
propostas e mostra, no monitoramento, o drift das propostas novas em relação à carteira (PSI de 0,376).
O desafio não foi implantado num Data Lake: é o desenho, com um protótipo local que roda.
Ver [`docs/arquitetura/camada_gold.md`](docs/arquitetura/camada_gold.md).

## Estrutura

| Pasta | O que tem |
|---|---|
| `src/autocred/` | biblioteca do projeto: leitura das bases, perda esperada, features, avaliação, faixas de score, modelos, escoragem, simulador de política e protótipo da camada Gold |
| `tests/` | 98 testes automáticos (pytest) |
| `notebooks/` | 01 diagnóstico, 02 torneio de modelos, 03 modelo final e 04 política, como fontes `.py` e notebooks executados `.ipynb` |
| `bases/` | as três bases fictícias, o dicionário de dados e os parâmetros de EAD e LGD |
| `site/` | a página interativa, em HTML, CSS e JavaScript, sem bibliotecas |
| `scripts/` | execução dos notebooks, dados da página, camada Gold, consulta ao Banco Central e scripts da entrega |
| `docs/` | desenho da camada Gold, especificações e planos de cada etapa, benchmarking e resultados oficiais |

## Como rodar

Requer Python 3.12 ou mais recente.

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Os notebooks levam cerca de 5 minutos (o torneio de modelos, uns 3) e gravam os resultados em `saidas/`:

```bash
bash scripts/rodar_notebooks.sh
```

Tudo usa semente fixa: rodando duas vezes, os resultados saem iguais.

Para regenerar os dados da página depois dos notebooks e abri-la no navegador:

```bash
python scripts/gerar_dados_site.py
python -m http.server 8000 --directory site
```

A página fica em `http://localhost:8000`.

Para montar a camada Gold com os resultados dos notebooks e rodar as consultas SQL de exemplo:

```bash
python scripts/gerar_camada_gold.py
```

`gerar_documento_politica.py` e `preparar_submissoes.py` produziram a entrega do desafio e dependem
de materiais da mentoria que não estão neste repositório. Ficam como registro.

## Dados

As bases são fictícias. Foram criadas pela mentoria JUMP para fins educacionais e de portfólio e não
têm dados de pessoas reais.

## Equipe (Grupo 2)

- Túlio Mazzarotto: modelo de PD
- Leandro Nogueira: política de crédito
- Fábio Piona de Sousa: defesa
- Joyce Marques Barbosa: apoio e revisão

Desafio da mentoria JUMP, da Analítica Educação.

## Uso de IA

Modelagem de dados, arquitetura de risco e premissas desenvolvidas pela equipe do projeto. Interface e
narrativa visual prototipadas em parceria com Claude Code.

## Licença

O código está sob a licença [MIT](LICENSE). As bases pertencem à mentoria JUMP e estão aqui para o fim
educacional para o qual foram criadas.
