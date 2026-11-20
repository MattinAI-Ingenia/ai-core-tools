-- Estimated cost of agent executions, from agent_execution_event tokens x pricing_catalog prices.
--   docker exec -i docker-postgres-1 psql -U mattin -d mattin_ai -v since="'2026-11-19 14:54'" -v until="'2026-11-19 16:10'" < scripts/agent_cost.sql
-- Rows with no price in the catalog (e.g. a self-hosted model) show their tokens under unpriced_tokens, not a cost.
-- Cached input uses cached_input_price_per_1m (input price if unknown); cached_input_tokens is NULL before revision cachetok001.
-- Events recorded before the query_specialists fix also hold their specialists' tokens: those are subtracted here.
WITH children AS (
    SELECT parent_execution_id AS pid, SUM(input_tokens) AS i, SUM(output_tokens) AS o, SUM(COALESCE(cached_input_tokens, 0)) AS c
    FROM agent_execution_event WHERE parent_execution_id IS NOT NULL AND started_at >= :since GROUP BY 1
), own AS (
    SELECT e.agent_id, e.model_name,
        CASE WHEN k.i IS NOT NULL AND e.input_tokens >= 0.9 * k.i THEN e.input_tokens - k.i ELSE e.input_tokens END AS input,
        CASE WHEN k.i IS NOT NULL AND e.input_tokens >= 0.9 * k.i THEN GREATEST(e.output_tokens - k.o, 0) ELSE e.output_tokens END AS output,
        CASE WHEN k.i IS NOT NULL AND e.input_tokens >= 0.9 * k.i THEN GREATEST(COALESCE(e.cached_input_tokens, 0) - k.c, 0) ELSE COALESCE(e.cached_input_tokens, 0) END AS cached
    FROM agent_execution_event e LEFT JOIN children k ON k.pid = e.event_id
    WHERE e.started_at >= :since AND e.started_at < :until AND e.input_tokens IS NOT NULL
)
SELECT o.agent_id, o.model_name, COUNT(*) AS executions,
    SUM(o.input) AS input_tokens, SUM(o.cached) AS cached_tokens, SUM(o.output) AS output_tokens,
    ROUND(SUM(CASE WHEN p.model_name IS NULL THEN 0 ELSE
        ((o.input - o.cached) * p.input_price_per_1m
         + o.cached * COALESCE(p.cached_input_price_per_1m, p.input_price_per_1m)
         + o.output * p.output_price_per_1m) / 1e6 END)::numeric, 4) AS cost_usd,
    SUM(CASE WHEN p.model_name IS NULL THEN o.input + o.output ELSE 0 END) AS unpriced_tokens
FROM own o LEFT JOIN pricing_catalog p ON p.model_name = o.model_name
GROUP BY GROUPING SETS ((o.agent_id, o.model_name), ())
ORDER BY o.agent_id NULLS LAST;
