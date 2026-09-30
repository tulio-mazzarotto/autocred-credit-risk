# AutoCred: PD model and credit policy

[![tests](https://github.com/tulio-mazzarotto/autocred-credit-risk/actions/workflows/testes.yml/badge.svg)](https://github.com/tulio-mazzarotto/autocred-credit-risk/actions/workflows/testes.yml)

*[Leia em português](README.md)*

Winning project of the AutoCred challenge, part of the JUMP mentorship by Analítica Educação, in
September 2026. It has two parts: a model that estimates the probability of default (PD) of car loan
applicants, and a credit policy built on top of it that decides who gets approved, at what rate, for
how long and with how much down payment.

**[Interactive page with the project story and a policy simulator](https://tulio-mazzarotto.github.io/autocred-credit-risk/?lang=en)**

## Result

The model was scored by AUC on a test set from a later period than the training data, and the policy
by annual ROI on a set of new applications. Both were measured against the real outcome, which we did
not have.

| Metric | Result | Board requirement |
|---|---:|---:|
| Model AUC (Gini) | **0.7711** (0.5422) | |
| Annual ROI | **16.53%** (forecast: 16.40%) | above 15% |
| Portfolio default rate | **4.75%** | at most 8% |
| Approval | **44.8%** | at least 35% |
| Contracted volume | **R$ 50.1M** | at least R$ 40M |
| Average rate | **2.51% per month** (63.9% of approved applicants signed) | at most 3.5% per month |

Source: the mentorship's official scoring on 26 Sep 2026, transcribed in [`docs/resultados_oficiais.json`](docs/resultados_oficiais.json).

## The problem

AutoCred is a fictional car finance company that lent cheaply to a risky clientele. Of the 10,000
contracts on record (2022 to 2024), 8.3% reached 90 days past due within the first year. The board
asked for a new policy with an annual ROI above 15%, within four limits:

- approve at least 35% of applications;
- default rate of at most 8% of the portfolio;
- at least R$ 40 million in contracts;
- interest of at most 3.5% per month.

The challenge datasets:

| Dataset | Rows | Use |
|---|---:|---|
| A | 10,000 contracts | model development, with the outcome |
| B | 3,000 contracts | model test (AUC) |
| C | 5,000 applications | policy test (ROI) |

Risk turns into price through the expected loss chain: **PD × EAD × LGD**. PD is the chance of not
paying; EAD, how much the customer owes at default; LGD, how much is lost after repossessing and
selling the car. The rate has to cover that loss and still leave a return.

## Decisions that made the difference

1. **A guard against data leakage.** The column `qtd_parcelas_em_atraso_12m` (late installments in the
   next 12 months) is only known after the loan is granted. On its own it reaches an AUC of 0.96 in
   training, and it is all zeros in the test sets. `dados.validar_sem_proibidas` blocks it, the target
   and any realized value from entering the model.
2. **Out-of-time validation.** Each candidate was trained on 2022 and tested on 2023, then trained on
   2022 and 2023 and tested on 2024. A more complex model only replaced a simpler one if it gained at
   least 0.005 of average AUC. Score distribution stability between the 2023 and 2024 vintages was
   also measured with the PSI (Population Stability Index): 0.0015, far below the 0.10 attention threshold.
3. **Monotonic constraints.** The winner, a LightGBM, is forbidden from learning relationships that make
   no business sense, such as more credit bureau restrictions lowering the risk. Average out-of-time AUC
   of 0.737, against 0.667 for logistic regression.
4. **Acceptance and adverse selection.** Not every approved applicant signs, and high rates attract the
   riskiest customers. The simulator (`politica.py`) projects three customer-reaction scenarios, and the
   chosen policy meets the limits even in the most severe one. Applications outside the historical
   profile have their PD doubled for prudence.
5. **Market-anchored pricing.** Rates range from 2.3% to 2.9% per month by score, compared with public
   Central Bank of Brazil data on 45 lenders that finance cars ([`docs/benchmark/`](docs/benchmark/)).

## Repository layout

| Folder | Contents |
|---|---|
| `src/autocred/` | project library: data loading, expected loss, features, evaluation, score bands, models, scoring and policy simulator |
| `tests/` | 92 automated tests (pytest) |
| `notebooks/` | 01 diagnosis, 02 model tournament, 03 final model and 04 policy, as `.py` sources and executed `.ipynb` notebooks |
| `bases/` | the three fictional datasets, the data dictionary and the EAD and LGD parameters |
| `site/` | the interactive page, in plain HTML, CSS and JavaScript |
| `scripts/` | notebook runner, page data, Central Bank queries and the challenge delivery scripts |
| `docs/` | specs and plans for each stage, benchmarking and official results |

The code, comments and notebooks are in Portuguese.

## How to run

Requires Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

The notebooks take about 5 minutes (the model tournament about 3) and write their results to `saidas/`:

```bash
bash scripts/rodar_notebooks.sh
```

Everything uses a fixed seed: running twice gives the same results.

To rebuild the page data after the notebooks and open it in a browser:

```bash
python scripts/gerar_dados_site.py
python -m http.server 8000 --directory site
```

The page is at `http://localhost:8000`.

`gerar_documento_politica.py` and `preparar_submissoes.py` produced the challenge delivery and depend
on mentorship materials that are not in this repository. They are kept as a record.

## Data

The datasets are fictional. They were created by the JUMP mentorship for education and portfolio use
and contain no data about real people.

## Team (Team 2)

- Túlio Mazzarotto: PD model
- Leandro Nogueira: credit policy
- Fábio Piona de Sousa: defense
- Joyce Marques Barbosa: support and review

A challenge from the JUMP mentorship by Analítica Educação.

## Use of AI

Data modeling, risk architecture and assumptions developed by the project team. Interface and visual
narrative prototyped in partnership with Claude Code.

## License

The code is under the [MIT](LICENSE) license. The datasets belong to the JUMP mentorship and are here
for the educational purpose they were created for.
