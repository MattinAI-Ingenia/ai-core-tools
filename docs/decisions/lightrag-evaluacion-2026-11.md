# Decisiones de la evaluación de LightRAG (noviembre 2026)

Decisiones tomadas durante la medición de tiempos, costes y modelos del GraphRAG
(LightRAG 1.5.6 en Mattin AI). El detalle de las medidas está en el documento
"GraphRAG (LightRAG): referencia técnica de tiempos, costes y modelos".

## Contenido

- **Edición vigente.** Cuando dos ediciones de un mismo documento se contradicen,
  la respuesta esperada y la del agente siguen la edición más reciente (fecha o número
  de edición más alto), y el agente dice cuál usa. La regla está en el system prompt de
  los agentes de evaluación de Domusa (agentes 10 a 29, regla 9). No se aplicó a los
  agentes 1 y 4.
- **Set de 136 PDF cerrado.** `benchmark/preguntas/es/eval_set_domusa_136.json`: las
  mismas 63 preguntas que `eval_set_domusa.json`, con respuestas y documentos esperados
  recalculados sobre los 136 PDF.

## Silo 55 (truncado)

- Su parte truncada (26 % de las extracciones cortadas en el tope de 8.192 tokens) no se
  usa como referencia. No se reindexa ni se hacen pasadas de evaluación sobre él.
- Los silos de referencia son el 57 (33 PDF) y el 59 (136 PDF).

## Pasadas de evaluación

- Las pasadas que se presentan se hacen sin reranker y sin el clasificador System One,
  que es lo que se entrega.
- Una pregunta que agota el límite de ejecución del agente (600 s) o supera el contexto
  del servidor cuenta como FAIL. Se anota aparte como timeout, pero no se excluye del
  porcentaje. Ejemplo: Qwen3.8-27B xhigh sobre el silo 59 queda en 26 PASS, 18 PARTIAL y
  19 FAIL (41,3 % PASS), de los que 17 son preguntas sin respuesta.

## Experimentos no prioritarios

- Desglose de tokens y latencia por fase (recuperación y síntesis): no prioritario.
- Comparar otros extractores (Qwen3-30B, Qwen3.6-35B-A3B) con el mismo set: no
  prioritario; ya hay un modelo on-prem y otro de pago.
- Validar el estimador de coste de OpenAI a 136 PDF: cerrado con nota; está calibrado con
  2 y 33 PDF y a esa escala no se valida por su coste.

## Mantenimiento

- Se borran de Neo4j los workspaces (`silo_N`) sin nodos y sin silo asociado en
  PostgreSQL, junto con sus índices. Los workspaces con nodos aunque su silo ya no exista
  (`silo_11`, `silo_12`, `silo_32`, `silo_33`, `silo_35`) no se tocan.
