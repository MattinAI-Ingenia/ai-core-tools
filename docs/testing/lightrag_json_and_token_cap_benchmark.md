# Benchmark: JSON forzado por esquema y tope de tokens en la extracción LightRAG

Mide el efecto de los dos cambios introducidos en la ruta de extracción —
`response_format` con esquema (`LIGHTRAG_EXTRACT_GUIDED_JSON`) y tope de salida
(`LIGHTRAG_EXTRACT_MAX_TOKENS`) — sobre el **mismo corpus** que
[`lightrag_extraction_benchmark_corpus.md`](lightrag_extraction_benchmark_corpus.md),
con análisis de estructura del grafo que aquel no incluía.

## Configuración

| | |
|---|---|
| Corpus | 4 manuales DOMUSA completos, 61 chunks de 4.000 caracteres (idéntico al benchmark de referencia) |
| Corpus secundario | 2 páginas de bibliografía de arXiv:2304.02381, reconstruidas del log de vLLM que motivó el tope |
| Modelos | `Qwen3-30B-A3B-Instruct` (vLLM propio) y `gpt-5.4-mini` (OpenAI) |
| Concurrencia | 4 chunks por documento, 4 documentos en paralelo (16 flujos), variantes en serie |
| Temperatura | 0.0 |
| Script | `scripts/compare_extraction.py`, que importa el esquema y el rescate **del propio adapter** |

Control de sanidad: `qwen_pre_change` da 1.627 entidades / 1.602 relaciones frente
a las 1.641 / 1.485 que reportó el benchmark de referencia para el mismo corpus.
Comparable.

## Volumen, coste y truncados

| Variante | Menc. | Únicas | Rel | out_tok | Cómputo | Chunks al tope | Rescatados | Perdidos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mini `json_object` | 1.148 | 660 | 974 | 92.024 | 455 s | 0 | 0 | 0 |
| mini esquema, tope 8192 | 996 | 593 | 782 | 74.533 | 393 s | 0 | 0 | 0 |
| Qwen prompt-only, sin tope | 1.627 | 1.025 | 1.602 | 223.696 | 4.596 s | 0 | 0 | 0 |
| Qwen esquema, tope 8192 | 1.676 | 1.050 | 1.454 | 212.812 | 4.530 s | 3 | 3 | 0 |
| Qwen esquema, tope 4096 | 1.644 | 1.022 | **919** | 171.261 | 3.631 s | 27 | 27 | 0 |

Tokens de salida por chunk (Qwen): mediana ~3.600, p90 ~6.400-6.900, máximo 12.504.

## Conclusiones

### 1. El tope de 8192 es prácticamente gratis

Se activa en **3 de 61 chunks (4,9%)** y no mueve las cifras: 1.676 entidades
frente a 1.627 sin tope, dentro de la varianza documentada de Qwen. El cómputo
tampoco baja (4.530 s vs 4.596 s), porque el 95% de los chunks nunca se acerca al
techo. Su valor no es el caso medio, es acotar el peor caso: sin tope hubo un
chunk de 12.504 tokens.

### 2. El tope de 4096 destruye el grafo sin que se note en las entidades

Se activa en **27 de 61 chunks (44%)** y produce el resultado más importante de
este benchmark: **las entidades sobreviven (1.644) pero las relaciones caen un
43% (919 vs 1.602)**. El motivo es el orden del JSON — el modelo emite primero
`entities` y después `relationships`, así que un corte se come siempre las
relaciones. Las aristas del grafo bajan de 1.481 a 850.

Un tope agresivo no "recorta un poco de todo": deja un grafo con las mismas
entidades y la mitad de los caminos. Ahorra 21% de tiempo a cambio de la mitad
del grafo. **No usar 4096.**

### 3. Sin el rescate, cada chunk al tope se perdía entero

Con `response_format` activo, el SDK de OpenAI **no devuelve texto truncado**:
lanza `LengthFinishReasonError`. Medido antes del arreglo: 2 de 17 chunks
perdidos por completo en un solo manual. `_salvage_length_limit`
(`adapters.py`) extrae el JSON parcial de dentro de la excepción y `json_repair`
recupera los registros completos. En esta corrida: **30 chunks tocaron el tope,
30 rescatados, 0 perdidos**.

### 4. El esquema no cuesta exhaustividad y limpia los tipos

En Qwen el esquema no penaliza nada (1.676 vs 1.627 entidades). En mini cuesta
**−10/−13% de entidades**, consistente en dos corridas independientes.

A cambio: `mini json_object` emite **27 entidades con tipos fuera del vocabulario**
(`concepto`, `objeto`, `organización` — traducciones al español de tipos que el
prompt pide en inglés); con esquema son **0**. El esquema declara `type` como
string libre, sin `enum`, así que es un efecto indirecto de la decodificación
restringida — y sugiere la mejora obvia: añadir el `enum` de tipos permitidos.

### 5. El grafo es radial en todas las configuraciones

| Variante | Grado mediano | Hojas | top-5 del grado | Aristas a hub | Aristas periféricas |
|---|---:|---:|---:|---:|---:|
| mini `json_object` | 1 | 52,8% | 18,5% | 36,4% | 4,6% |
| mini esquema | 1 | 54,3% | 18,8% | 36,8% | 5,2% |
| Qwen sin tope | 1 | 56,6% | 16,0% | 31,9% | 4,5% |
| Qwen tope 8192 | 1 | 59,7% | 16,6% | 32,9% | 7,0% |

**Grado mediano 1 y más de la mitad de nodos son hojas.** Un 32-37% de las
aristas tocan uno de los cinco hubs y solo un **4,5-7% unen dos nodos
periféricos**. Casi cualquier camino de dos saltos pasa por un hub.

Los hubs son siempre los nombres de producto (`jaka hfd condens`, `bt duo`,
`terma hv`, `mcf hdx/hdn`) más tres genéricos (`quemador`, `caldera`,
`domusa teknik`). En Qwen `quemador` llega a 158 vecinos distintos frente a 61 en
mini.

**Qwen no compra tejido, compra hojas**: 1,55× entidades que mini con la misma
proporción de aristas periféricas (4,5% vs 4,6%). Para RAG multi-hop, más
entidades colgando del mismo hub no añaden caminos nuevos.

### 6. Fragmentación de entidades

Pares donde un nombre contiene a otro y LightRAG (que fusiona solo por nombre
exacto) los deja separados: **70-72 en mini, 127-145 en Qwen**. Escala con el
volumen. Ejemplos: `'bt duo'` (grado 64) ⊂ `'tapa elíptica bt duo 500-1000'`
(grado 1); `'mcf 40'` ⊂ `'quemador mcf 40'`. Las variantes largas casi siempre
tienen grado 0-3, así que son nodos ruidosos más que hubs partidos.

### 7. Coste relativo de Qwen

**10,1× el cómputo de mini para 1,55× las entidades** (4.596 s vs 455 s). El
benchmark de referencia estimaba ~4×; con un solo modelo ocupando la GPU la
diferencia es mayor.

## Corpus patológico: una página de bibliografía

| Variante | Ent | Rel | out_tok | Tiempo | Al tope |
|---|---:|---:|---:|---:|---|
| Qwen sin tope | 84 | 68 | 10.845 | 109 s | no |
| Qwen tope 8192 | 84 | 24 | 7.548 | 76 s | no |
| Qwen tope 4096 | 66 | **0** | 4.096 | 41 s | sí, rescatado |
| gpt-5.4-mini (referencia) | 39 | 29 | 2.842 | 14 s | no |
| gpt-5.4-nano (referencia) | 33 | 26 | 2.440 | 21 s | no |

Qwen extrae **84 entidades y 68 relaciones de una lista de citas** donde los
modelos cloud encuentran ~35 y ~28. Gasta 10.845 tokens y 109 s en una sola
página: 4× el tiempo de mini. No es un fallo de formato, es densidad inventada —
coautorías entre nombres de una bibliografía.

El tope de 4096 sobre esa página deja **0 relaciones**, coherente con el punto 2.

## Recomendaciones

1. **`LIGHTRAG_EXTRACT_MAX_TOKENS=8192`**: mantener. Acota el peor caso sin coste
   medible.
2. **No bajar a 4096**: destruye el 43% de las relaciones de forma invisible en
   el recuento de entidades.
3. **`LIGHTRAG_EXTRACT_GUIDED_JSON=true`**: mantener. Gratis en Qwen, elimina los
   tipos fuera de vocabulario, y el coste en mini (−10%) se compensa con eso.
4. **`enum` de tipos en el esquema: descartado.** Convertiría la garantía
   empírica en estructural, pero el fallo que arreglaría (tipos en español) solo
   apareció en `mini` + `json_object` — 27 entidades de 1.148, 2,4% — y esa
   configuración ya no es la de producción: con esquema, `mini` emite 0 y **Qwen
   emite 0 incluso sin esquema** (ver la tabla de tipos, §4). Además los tipos son
   configurables por silo (`silo.lightrag_entity_types`, que LightRAG consume como
   texto libre en `addon_params['entity_types_guidance']`), así que el `enum`
   habría que derivarlo de esa guía o duplicar la lista: si se desincronizan, el
   esquema rechaza un tipo que el prompt sí permite y rompe la extracción entera.
   Cambia un problema del 2,4% por un modo de fallo del 100%.
5. **Pendiente**: normalizar nombres antes de fusionar, o al menos medirlo: 127-145
   pares subsumidos por corpus es ruido estructural que LightRAG no resuelve.
6. **Sobre el modelo de extracción**: si el criterio es calidad de grafo por euro
   de cómputo, mini gana con claridad en este corpus. Qwen aporta exhaustividad
   real (+55% entidades) pero en forma de hojas, no de conectividad.

## Reproducir

```bash
# Los resultados crudos quedan en backend/data/bench/ (volumen montado)
docker compose exec backend python /aict_backend/data/bench/compare_extraction.py \
  /aict_backend/data/bench/data/CDOC001004.pdf \
  -m "openai:Qwen3-30B-A3B-Instruct@http://<host>:8095/v1" \
  --json --response-format schema --max-tokens 8192 --max-chunks 999 --out /tmp/x

# response-format: none (solo prompt) | object (json_object) | schema (produccion)
# max-tokens 0 = sin tope
```

## Advertencias metodológicas

- **Una corrida por variante.** Qwen tiene varianza real entre pasadas idénticas
  con `temperature=0`: sobre un solo documento medí 376, 383, 483 y 496 entidades
  para configuraciones equivalentes (±14%). Por eso todas las conclusiones se
  sacan de los 61 chunks y no de documentos sueltos; diferencias menores del ~15%
  en volumen no son concluyentes. Las que sí lo son (relaciones −43% con tope
  4096, tipos fuera de vocabulario, chunks al tope) están muy por encima de ese
  ruido.
- Las latencias se midieron con un solo modelo ocupando la GPU y 16 flujos
  concurrentes; no son comparables con las del benchmark de referencia, que
  repartía la GPU entre dos modelos locales.
