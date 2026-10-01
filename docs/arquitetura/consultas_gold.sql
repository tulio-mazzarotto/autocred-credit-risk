-- Consultas de exemplo sobre a camada Gold do AutoCred.
-- Escritas para o Databricks SQL (Spark SQL); os testes também as executam num SQLite em memória,
-- com as tabelas no esquema "gold". A política em vigor é a linha com vigente_ate nulo.

-- consulta: motor_de_decisao_em_lote
-- Motor de decisão em lote: cruza cada proposta escorada com a regra de preço em vigor para o score dela.
WITH avaliadas AS (
  SELECT
    p.id_proposta,
    p.score_1a10,
    p.prazo_desejado_meses,
    r.versao_politica,
    r.aprovar,
    r.taxa_am,
    r.prazo_max_meses,
    r.entrada_min_pct,
    (p.fora_perfil AND NOT r.aceita_fora_perfil) AS barrada_pelo_perfil
  FROM gold.propostas_escoradas AS p
  JOIN gold.regras_precificacao AS r
    ON r.score_1a10 = p.score_1a10
   AND r.vigente_ate IS NULL
)
SELECT
  id_proposta,
  score_1a10,
  CASE WHEN aprovar AND NOT barrada_pelo_perfil THEN 'APROVAR' ELSE 'NEGAR' END AS decisao,
  CASE
    WHEN NOT aprovar THEN 'score abaixo do corte'
    WHEN barrada_pelo_perfil THEN 'fora do perfil'
    ELSE 'aprovada'
  END AS motivo,
  CASE WHEN aprovar AND NOT barrada_pelo_perfil THEN taxa_am END AS taxa_am,
  CASE WHEN aprovar AND NOT barrada_pelo_perfil THEN
    CASE WHEN prazo_desejado_meses < prazo_max_meses THEN prazo_desejado_meses ELSE prazo_max_meses END
  END AS prazo_ofertado_meses,
  CASE WHEN aprovar AND NOT barrada_pelo_perfil THEN entrada_min_pct END AS entrada_min_pct,
  versao_politica
FROM avaliadas
ORDER BY id_proposta;

-- consulta: painel_aprovacao_por_canal
-- Painel operacional: volume de propostas, aprovação e taxa média ofertada por canal de originação.
SELECT
  canal_originacao,
  COUNT(*) AS propostas,
  AVG(CASE WHEN decisao = 'APROVAR' THEN 1.0 ELSE 0.0 END) AS taxa_aprovacao,
  AVG(taxa_am) AS taxa_media_am
FROM gold.decisoes_credito
GROUP BY canal_originacao
ORDER BY propostas DESC;

-- consulta: painel_por_score
-- Painel operacional: como as propostas se distribuem pela régua e o que a política fez em cada faixa.
SELECT
  score_1a10,
  COUNT(*) AS propostas,
  SUM(CASE WHEN decisao = 'APROVAR' THEN 1 ELSE 0 END) AS aprovadas,
  SUM(CASE WHEN fora_perfil THEN 1 ELSE 0 END) AS fora_do_perfil,
  AVG(taxa_am) AS taxa_media_am
FROM gold.decisoes_credito
GROUP BY score_1a10
ORDER BY score_1a10;

-- consulta: monitoramento_psi
-- Risco de modelo: PSI de cada população contra a referência, com a leitura usual dos limites.
SELECT
  populacao,
  referencia,
  SUM(contribuicao_psi) AS psi,
  CASE
    WHEN SUM(contribuicao_psi) >= 0.25 THEN 'drift severo'
    WHEN SUM(contribuicao_psi) >= 0.10 THEN 'atenção'
    ELSE 'estável'
  END AS situacao
FROM gold.monitoramento_score
WHERE populacao <> referencia
GROUP BY populacao, referencia
ORDER BY populacao;
