# Benchmark de extracción de entidades LightRAG — corpus DOMUSA TEKNIK (4 manuales completos)

Comparativa de calidad/latencia entre modelos cloud (OpenAI) y open-source
autoalojados (Qwen3-30B-A3B-Instruct, Mistral-Small-3.2-24B-Instruct, ambos
vía vLLM) en los dos roles de LLM que usa LightRAG en este repo:
**`extract`** (entidades/relaciones sobre chunks del documento, indexación —
`LIGHTRAG_EXTRACT_MODEL`), sobre un corpus de 4 documentos completos, y
**`keyword`** (high/low-level keywords de una consulta de usuario, en tiempo
de query — `LIGHTRAG_KEYWORD_MODEL`), sobre el primero de esos documentos.
Ejecutada con
[`scripts/compare_extraction.py`](../../scripts/compare_extraction.py), que
reutiliza los prompts reales de `lightrag-hku` en vez de una aproximación.

## Alcance: qué mide este benchmark y qué no

El script llama al LLM **exactamente igual que LightRAG en producción**
para cada rol — mismos prompts, mismo formato de entrada y salida, mismos
parámetros — pero solo aísla esa llamada. No reproduce el resto del pipeline
de indexación/consulta:

- **Gleaning**: LightRAG puede reejecutar la extracción sobre el mismo chunk
  con un prompt "continue" (`entity_extract_max_gleaning`) para pescar
  entidades que se le escaparon a la primera. El script llama al LLM una
  sola vez por chunk.
- **Fusión de entidades y relaciones entre chunks**: cuando una entidad
  aparece en varios chunks (p. ej. "BT DUO" en el chunk 0 y en el chunk 4),
  LightRAG la fusiona en un único nodo del grafo, y puede volver a llamar al
  LLM para resumir la descripción combinada
  (`summarize_entity_descriptions`). Este script cuenta cada chunk por
  separado — los duplicados y la estructura hub-and-spoke que se documentan
  abajo son **previos a esa fusión**; el paso de fusión podría reducirlos,
  amplificarlos o dejarlos igual, y no se ha medido.
- **Grafo, embeddings y comunidades**: no se construye nada en Neo4j, no se
  generan embeddings de entidades/relaciones, no se ejecuta detección de
  comunidades (Leiden) — todo lo que usan los modos `local`/`global` de
  recuperación queda fuera de este benchmark.
- **Troceo por tokens**: LightRAG trocea por tokens (tiktoken, con
  solapamiento configurable); este script trocea por caracteres — una
  simplificación deliberada, vale porque para comparar modelos entre sí solo
  importa que el chunk sea idéntico para todos, no que coincida con el
  troceo exacto de producción.

En resumen: este benchmark responde a "¿qué modelo da mejor extracción
cruda por chunk / mejores keywords por consulta?" — el cuello de botella
real para elegir modelo. No responde a "¿qué grafo final queda tras indexar
el documento completo en producción?".

## Corpus de prueba

Los 4 PDF de `data/` en la raíz del repo, todos manuales DOMUSA TEKNIK
(instalación y funcionamiento), completos (no solo un extracto):

| Documento | Producto | Páginas | Caracteres | Chunks (4.000 car.) |
| --- | --- | ---: | ---: | ---: |
| `CDOC000810.pdf` | BT DUO — depósito de inercia con A.C.S. integrado | 24 | 35.805 | 9 |
| `CDOC001004.pdf` | Caldera de calefacción MCF HDX / MCF HDN (gasóleo) | 36 | 67.098 | 17 |
| `CDOC001009.pdf` | Caldera de calefacción JAKA HFD CONDENS (gasóleo, condensación) | 36 | 73.497 | 19 |
| `CDOC001048.pdf` | Caldera de calefacción TERMA HV (gasóleo) | 36 | 60.095 | 16 |
| **Total** | | **132** | **236.495** | **61** |

Mismo fabricante y estructura documental (advertencias de seguridad,
instalación hidráulica/eléctrica, características técnicas, esquemas,
mantenimiento, listados de repuesto), pero productos distintos (un
acumulador vs. tres calderas de gasóleo) — permite comprobar si los patrones
observados en el benchmark de un solo documento son del modelo o del
documento.

## Configuración

Idéntica a la del benchmark de referencia, con dos diferencias por el
volumen (4 documentos × 4 modelos en paralelo):

| Parámetro | Valor | Motivo |
| --- | --- | --- |
| Modo de extracción | `--json` (`ENTITY_EXTRACTION_USE_JSON=true`) | Imprescindible con modelos open-source (ver más abajo) |
| `--max-tokens` | 10.000 | Compromiso único para los 4 modelos a la vez; Mistral-Small tiene solo 16.384 de contexto |
| Concurrencia | 4 chunks simultáneos por modelo (default, = `DEFAULT_MAX_ASYNC` de LightRAG) | Los 4 documentos se lanzaron **en paralelo** entre sí además — 16 flujos concurrentes reales contra los 2 endpoints propios (Qwen, Mistral) |
| Temperatura | 0.0 | Igual que en producción |

### Por qué `--json` es obligatorio

El repo **no activa** `ENTITY_EXTRACTION_USE_JSON` en ningún sitio — LightRAG
corre por defecto en modo texto delimitado (`entity<|#|>nombre<|#|>...`).
Antes de este benchmark se probó ese modo por defecto sobre BT DUO
(20 páginas, 1 chunk) y falló de dos formas distintas, una por modelo:

- **Qwen3-30B-A3B**: no respetaba el límite del propio prompt
  (`max_total_records=100`) y nunca emitía `<|COMPLETE|>` — generó **581
  relaciones**, la mayoría inventadas (*"BT DUO → debe incluir → sección de
  mantenimiento"* repetido para cada línea del índice), llenando los 32k de
  contexto disponibles y cortándose a media palabra (310 s por chunk). En
  modo `--json` sobre el mismo chunk: 20 relaciones, todas verificables.
- **gpt-5.4-nano**: en un chunk denso (vaciado + características técnicas +
  esquema eléctrico + repuestos), dejó de repetir el prefijo
  `entity<|#|>`/`relation<|#|>` y los saltos de línea entre registros a
  partir del segundo — colapsando ~20 entidades reales y varias relaciones
  en un único bloque de 15.692 caracteres **sin ningún salto de línea**, del
  que el parser (idéntico al de LightRAG) solo puede recuperar 1 entidad con
  una descripción-basura y 0 relaciones. Sin error visible: el modelo
  terminó con `finish_reason=stop`, no truncó por tokens — el formato se
  rompió antes, en silencio. En modo `--json` ese mismo chunk dio 30
  entidades y 28 relaciones válidas.

Ninguno de los dos es un fallo de capacidad del modelo: Qwen no sigue bien
un límite numérico en texto libre; nano no mantiene un formato repetitivo
frágil en contenido muy repetitivo. El esquema JSON no permite ninguno de
los dos fallos por construcción (cada entidad es un objeto de un array, no
depende de que el modelo repita un token literal en el momento correcto) —
por eso `--json` no es un parche solo para modelos pequeños/open-source,
sino una mitigación de una clase de pérdida silenciosa de datos que afecta
también a un modelo cloud.

Comando (uno por documento, en paralelo):

```bash
for f in CDOC000810 CDOC001004 CDOC001009 CDOC001048; do
  python scripts/compare_extraction.py "data/${f}.pdf" \
    -m openai:gpt-5.4-mini \
    -m openai:gpt-5.4-nano \
    -m "openai:Qwen3-30B-A3B-Instruct@http://<qwen-vllm-host>:8095/v1" \
    -m "openai:Mistral-Small-3.2-24B-Instruct@http://<mistral-vllm-host>:8096/v1" \
    --json --max-chunks 999 --max-tokens 10000 --out "graph-bench/${f}" &
done
wait
```

## Modelos comparados

Los mismos 4 del benchmark de referencia:

- **`gpt-5.4-mini`** y **`gpt-5.4-nano`** (OpenAI, cloud).
- **`Qwen3-30B-A3B-Instruct-2507`** (open-source, MoE 30B/3B-activos, vLLM propio).
- **`Mistral-Small-3.2-24B-Instruct-2506`** (open-source, denso 24B, vLLM propio, contexto 16.384).

## Resultados agregados (61 chunks, 4 documentos)

| Modelo | Entidades | Relaciones | Tipos de entidad usados | Tokens de salida | Tiempo (suma) |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-5.4-mini | 1.213 | 1.069 | 12 | 94.227 | 513 s |
| gpt-5.4-nano | 1.599 | 1.297 | 11 | 167.182 | 1.138 s |
| Qwen3-30B-A3B-Instruct | 1.641 | 1.485 | 11 | 211.712 | 4.548 s |
| Mistral-Small-3.2-24B-Instruct | 1.241 | 969 | 11 | 124.872 | 2.929 s |

Confirma la tendencia del benchmark de un documento: **Qwen y nano son los
más exhaustivos** (más entidades y relaciones), **mini es el más selectivo y
barato**, **Mistral queda en un punto intermedio** en volumen pero con el
`out_tok`/relación más alto de los cuatro (menos denso por token generado).

### Por documento

| Documento | mini (ent/rel) | nano (ent/rel) | Qwen (ent/rel) | Mistral (ent/rel) | Tiempo de pared (4 modelos, concurrencia=4) |
| --- | --- | --- | --- | --- | ---: |
| CDOC000810 (BT DUO) | 141 / 129 | 216 / 175 | 214 / 192 | 147 / 131 | 197 s |
| CDOC001004 (MCF HDX/HDN) | 413 / 371 | 510 / 375 | 508 / 428 | 417 / 326 | 387 s |
| CDOC001009 (JAKA HFD CONDENS) | 321 / 277 | 484 / 397 | 488 / 444 | 339 / 226 | 390 s |
| CDOC001048 (TERMA HV) | 338 / 292 | 389 / 350 | 431 / 421 | 338 / 286 | 338 s |

El ranking de volumen (Qwen ≳ nano > Mistral ≳ mini) se mantiene documento a
documento — no es un efecto de un documento concreto.

**Tiempo real de pared del lote completo (4 documentos en paralelo, 16 flujos
concurrentes): ~7 min 7 s**, frente a los ~22 min que habría costado la suma
de los cuatro tiempos de pared individuales si se hubieran corrido en serie —
paralelizar por documento (no solo por chunk dentro de un documento) es una
ganancia real cuando hay varios documentos que indexar, incluso compartiendo
los mismos dos servidores vLLM.

## Páginas en blanco: los 4 modelos las manejan bien

Los cuatro manuales terminan en varias páginas de "NOTAS" completamente en
blanco (solo puntos de relleno para escribir a mano). Varios chunks del
corpus caen íntegramente en esas páginas — 8 de los 61 chunks, repartidos
entre los 4 documentos. En **todos los casos y con los 4 modelos**, el
resultado fue `{"entities": [], "relationships": []}` en <1 s, sin ningún
error ni contenido inventado. Ninguno de los 4 modelos alucinó entidades a
partir de una página vacía — un resultado positivo y uniforme, sin
diferencias entre cloud y open-source aquí.

## Calidad de entidades y relaciones

### El patrón "hub-and-spoke": corregido con los 61 chunks

⚠️ **Corrección respecto a una versión anterior de este documento.** La
primera medición se hizo sobre **un solo chunk** (la introducción de BT DUO)
y llevaba a una conclusión equivocada: "Qwen y Mistral no generan relaciones
fuera del hub (0%)". Con muestra de 1, eso es ruido, no un patrón — los
chunks introductorios centran casi todo en el nombre del producto para
*cualquier* modelo, así que un solo chunk de ese tipo no dice nada sobre el
comportamiento general.

Con los datos reales — las 969-1.485 relaciones que genera cada modelo sobre
las 61 chunks del corpus, midiendo por chunk cuál es el nodo más citado (el
"hub" que cada modelo elige, no un nombre fijo) y qué fracción de relaciones
de ese chunk no lo tocan:

| Modelo | Relaciones no-hub | Relaciones totales | % no-hub |
| --- | ---: | ---: | ---: |
| Mistral-Small-3.2-24B | 367 | 969 | 37.9% |
| Qwen3-30B-A3B-Instruct | 713 | 1.485 | 48.0% |
| gpt-5.4-mini | 514 | 1.069 | 48.1% |
| gpt-5.4-nano | 817 | 1.297 | **63.0%** |

Qwen y mini están prácticamente empatados (~48%); Mistral es el más
centrado en el hub del chunk (38%, no 0%); nano sigue siendo el más
distribuido pero por un margen mucho menor del que sugería la muestra de 1
chunk (63% frente al 71% original, no frente a un 0%). Ningún modelo es
incapaz de generar relaciones no-hub — la diferencia es de grado, no de
capacidad, y una parte de la variación real es efecto del contenido del
chunk (introducción vs. sección técnica densa), no solo del modelo.

**Ni siquiera 8-9 chunks bastan para fiarse del número.** Repitiendo la
misma medición pero limitada a un solo documento del corpus (CDOC000810 —
BT DUO completo, 24 páginas, 8-9 chunks por modelo en vez de 61), el orden
relativo **cambia**:

| Modelo | % no-hub (solo CDOC000810, 8-9 chunks) | % no-hub (corpus completo, 61 chunks) |
|---|---:|---:|
| Qwen3-30B-A3B-Instruct | 29.2% | 48.0% |
| gpt-5.4-mini | 38.0% | 48.1% |
| Mistral-Small-3.2-24B | 48.9% | 37.9% |
| gpt-5.4-nano | 60.0% | 63.0% |

Mistral pasa de ser el más centrado en el hub (corpus completo) al segundo
*menos* centrado (un solo documento) — un cambio de posición, no solo de
magnitud. Qwen se mueve casi 19 puntos entre ambas mediciones. Solo nano se
mantiene estable y en el extremo "más distribuido" en las tres mediciones
(1 chunk, 8-9 chunks, 61 chunks). Conclusión práctica: **para comparar
modelos en este eje hace falta el agregado de varios documentos** — el
número de un documento suelto, incluso con varios chunks, no es
representativo.

### Contenido legal/regulatorio: los 4 modelos, de acuerdo

CDOC001009 incluye una cláusula legal poco común en el resto del corpus
(gestión ambiental de residuos de envases), con dos referencias normativas
específicas: **Ley 11/1997** y **Real Decreto 782/1998**. Los 4 modelos —
cloud y open-source, caro y barato — las extrajeron correctamente como
entidades `Concept`, con el mismo nombre exacto y sin errores. Sobre
contenido factual concreto y verificable (una cita legal con número y año),
no hay diferencia de calidad entre modelos; las diferencias de granularidad
y estructura documentadas arriba aparecen en cómo se organiza el
conocimiento, no en si los hechos concretos se capturan bien.

## Coste / latencia

- El ranking de velocidad se mantiene: mini y nano (cloud) muy por delante de
  Qwen y Mistral (self-hosted) en tiempo por chunk, consistente con el
  benchmark de un documento.
- **Qwen es el más caro en tiempo de cómputo total** (4.548 s sumados, ~4×
  mini) pero también el más exhaustivo — la relación entidades-por-segundo
  no es mala, es que hace mucho más trabajo por chunk.
- Mistral generó menos entidades que Qwen con un tiempo de cómputo similar
  (2.929 s) — peor ratio exhaustividad/latencia de los cuatro en este corpus.
- El cuello de botella real para *este* benchmark no fueron los modelos cloud
  ni el volumen de chunks, sino el límite de contexto de Mistral (16.384):
  obligó a fijar `--max-tokens` más bajo (10.000) para los 4 modelos a la
  vez, en vez del valor más generoso que se pudo usar en el benchmark de un
  documento cuando se probaba cada modelo por separado.

## Rol `keyword`: high/low-level keywords de consulta

A diferencia de `extract`, este rol no opera sobre el documento sino sobre
la **consulta del usuario** en tiempo de query (`LIGHTRAG_KEYWORD_MODEL`,
`extract_keywords_only` en `operate.py`): extrae `high_level_keywords`
(tema/intención) y `low_level_keywords` (entidades y detalles concretos)
para alimentar la recuperación local/global. El prompt (`keywords_extraction`)
manda **siempre** `response_format={"type": "json_object"}`, sin variante de
texto delimitado y sin system prompt — un único mensaje.

⚠️ **Alcance:** probado solo con preguntas sobre `CDOC000810` (BT DUO), no
repetido sobre los otros 3 documentos del corpus — el volumen de trabajo no
lo justificaba (7 preguntas ya bastan para medir cumplimiento de esquema y
latencia; no hay "chunks" que trocear en este rol).

### Preguntas de prueba

7 preguntas sobre el manual BT DUO, con distinta mezcla de intención
(high-level) y detalle concreto (low-level), más un caso límite que el propio
prompt manda tratar con listas vacías:

| # | Pregunta |
| --- | --- |
| 0 | ¿Cuál es la presión máxima de trabajo del circuito primario del BT DUO? |
| 1 | ¿Qué anticongelante recomienda DOMUSA TEKNIK para el circuito primario y por qué? |
| 2 | ¿Cuáles son las diferencias de volumen total y de A.C.S. entre el BT Duo 150 y el BT Duo 1000? |
| 3 | ¿Cómo se debe conectar eléctricamente el BT DUO a una caldera BioClass NG frente a una Lignum IB? |
| 4 | ¿Qué medidas de seguridad hay que tener en cuenta al instalar y mantener el BT DUO? |
| 5 | ¿Qué código de repuesto corresponde al termostato de control del BT Duo 150? |
| 6 | `hola` (caso límite: el prompt exige `{"high_level_keywords": [], "low_level_keywords": []}`) |

Se probaron 2 modelos open-source adicionales aquí, ambos servidos en su
propio vLLM y configurados en distintos momentos como
`LIGHTRAG_KEYWORD_URL`/`LIGHTRAG_KEYWORD_MODEL` en el `.env` del backend:

- **`Mistral-Small-3.2-24B-Instruct-2506`** (`--max-model-len 16384`) —
  retirado después: no cabía en VRAM junto al resto de modelos servidos a
  la vez. Los resultados se mantienen abajo, es el mismo modelo probado.
- **`Qwen3-4B-Instruct-2507`** (`--max-model-len 32768`) — el candidato
  actual, mucho más pequeño (4B) que el resto de modelos de este benchmark.

### Resultados (rol `keyword`)

| Modelo | HL keywords (total) | LL keywords (total) | Esquema inválido | Tiempo (suma) |
| --- | ---: | ---: | ---: | ---: |
| gpt-5.4-nano | 22-23 | 17-18 | 0/7 | 7.7-12.5 s |
| Qwen3-30B-A3B-Instruct | 20 | 9 | 0/7 | 4.1-4.8 s |
| Qwen3-4B-Instruct | 17 | 9 | 0/7 | 16.0 s |
| Mistral-Small-3.2-24B-Instruct | 14 | 9 | 0/7 | 13.2 s |

(Rangos: cada modelo se corrió en dos pasadas distintas — una con Mistral,
otra con Qwen3-4B — así que la tabla junta ambas; los conteos por modelo no
cambian entre pasadas, la latencia sí tiene la variación normal de red.)

Los cuatro: **100% de cumplimiento de esquema** (siempre exactamente
`high_level_keywords`/`low_level_keywords`, ambas listas de strings) y
respetan la regla de listas vacías en `hola`. A diferencia del rol `extract`,
aquí **la latencia no la domina el modelo cloud**: Qwen3-30B-A3B es
consistentemente el más rápido (0.2-0.9 s/consulta); Qwen3-4B es lento en
las primeras consultas (3.6-3.8 s, arranque en frío del servidor) y luego
tan rápido como los demás (0.2-0.45 s) — mismo patrón que mostró Mistral. La
salida es minúscula en los cuatro casos, así que la latencia la domina el
*round-trip* de red y el arranque del servidor, no la generación — justo lo
contrario que en `extract`.

**Qwen3-4B-Instruct es el hallazgo más interesante de esta ronda**: con solo
4B parámetros, **cero fallos** en las 7 preguntas — ni la alucinación de
Qwen3-30B ("sistema de enfriamiento") ni la verbosidad de gpt-5.4-nano.
Keywords tan concisas como las de Mistral, con mejor cobertura
(17 LL-relevantes vs. las mismas 9 de Mistral en LL, aunque menos HL: 17 vs.
20-23 de los modelos grandes). Para una tarea tan acotada como extraer
keywords de una consulta corta, el tamaño del modelo no está correlacionado
con la calidad observada en este test — ver comparación cualitativa.

### Comparación cualitativa (rol `keyword`)

```text
Pregunta 0: ¿Cuál es la presión máxima de trabajo del circuito primario del BT DUO?

gpt-5.4-nano:   HL=[presión máxima de trabajo; circuito primario; BT DUO]
                LL=[presión máxima de trabajo del circuito primario del BT DUO;
                     circuito primario del BT DUO]

Qwen3-30B-A3B:  HL=[presión máxima de trabajo; circuito primario; sistema de enfriamiento]
                LL=[BT DUO]

Qwen3-4B:       HL=[presión máxima de trabajo; circuito primario]
                LL=[BT DUO]

Mistral-Small:  HL=[presión máxima de trabajo; circuito primario]
                LL=[BT DUO]
```

Cuatro fallos posibles, tres modelos con alguno, uno sin ninguno:

- **Qwen3-30B** introduce **"sistema de enfriamiento"** como keyword de alto
  nivel — el BT DUO es un depósito de *calefacción*/A.C.S., no de
  refrigeración; no aparece en la pregunta ni tiene soporte en el contexto.
  El prompt lo prohíbe explícitamente (regla 4, *"Source of Truth: ... Do
  not infer unsupported facts"*). Puntual, no se repite en las otras 6
  preguntas.
- **gpt-5.4-nano** viola la regla 5 (*"Concise & Meaningful"*): sus
  `low_level_keywords` son casi la pregunta entera repetida
  ("presión máxima de trabajo del circuito primario del BT DUO" como
  keyword, no como pregunta), en vez de términos concretos y cortos.
- **Qwen3-4B y Mistral-Small**: ninguno de los dos fallos en esta pregunta —
  ver el resto de ejemplos abajo para ver si se mantiene.

```text
Pregunta 3: ¿Cómo se debe conectar eléctricamente el BT DUO a una caldera BioClass NG frente a una Lignum IB?

gpt-5.4-nano:   HL=[conexión eléctrica; BT DUO; caldera; BioClass NG; Lignum IB]
                LL=[BT DUO; BioClass NG; Lignum IB; caldera BioClass; caldera Lignum]

Qwen3-30B-A3B:  HL=[conexión eléctrica; instalación; comparación de modelos; funcionamiento de calderas]
                LL=[BT DUO; caldera BioClass NG; Lignum IB]

Qwen3-4B:       HL=[conexión eléctrica; caldera BioClass NG; Lignum IB]
                LL=[BT DUO; BioClass NG; Lignum IB]

Mistral-Small:  HL=[conexión eléctrica; caldera; comparación]
                LL=[BT DUO; BioClass NG; Lignum IB]
```

Aquí se repite y se ve más claro el fallo de nano: `caldera BioClass` y
`caldera Lignum` en `low_level_keywords`, **además** de `BioClass NG` y
`Lignum IB` ya listados — 4 keywords para 2 conceptos, redundancia real, no
solo verbosidad puntual. Los otros tres (Qwen3-30B, Qwen3-4B, Mistral) dan
listas limpias y sin solapamiento entre sí — la pregunta es genuinamente
fácil para los tres, Qwen3-4B entre ellos sin perder nada.

```text
Pregunta 4: ¿Qué medidas de seguridad hay que tener en cuenta al instalar y mantener el BT DUO?

gpt-5.4-nano:   HL=[medidas de seguridad; instalación; mantenimiento; BT DUO]
                LL=[instalar y mantener el BT DUO; BT DUO]

Qwen3-30B-A3B:  HL=[medidas de seguridad; instalación; mantenimiento]
                LL=[BT DUO]

Qwen3-4B:       HL=[medidas de seguridad; instalación; mantenimiento]
                LL=[BT DUO]

Mistral-Small:  HL=[medidas de seguridad; instalación y mantenimiento]
                LL=[BT DUO]
```

Confirma el patrón de nano: `low_level_keywords=["instalar y mantener el BT
DUO", "BT DUO"]` — la primera keyword es casi la pregunta reformulada, no
una entidad concreta. Se repite ya en 3 de las 7 preguntas (0, 3, 4), así
que no es puntual — es un rasgo consistente de cómo nano interpreta la
regla de concisión con este prompt. **Qwen3-30B y Qwen3-4B dan la
respuesta idéntica** en esta pregunta — a esta escala de tarea, el modelo
30B no aporta nada que el 4B no dé ya.

En las preguntas 1, 2 y 5 (no mostradas) los cuatro modelos coinciden en lo
esencial — mismas entidades de bajo nivel, keywords de alto nivel
sinónimas ("recomendación de anticongelante" vs. "anticongelante" —
diferencia de fraseo, no de contenido). Los fallos anteriores son la
excepción, no la norma: de 7 preguntas × 4 modelos = 28 respuestas, solo 4
tienen algún defecto (1 alucinación de Qwen3-30B, 3 de verbosidad/
redundancia de nano) — **Qwen3-4B y Mistral-Small no tuvieron ningún
defecto en ninguna de las 7 preguntas**.

## Conclusiones

1. El ranking de volumen (Qwen ≳ nano > Mistral ≳ mini) y la necesidad de
   `--json` **se sostienen en un corpus 6× más grande y con contenido
   variado** — no son artefactos de un documento concreto. La estructura del
   grafo (relaciones no-hub) también varía por modelo, pero solo es fiable
   medida con muestras grandes (61 chunks) — con menos, hasta el orden
   relativo entre modelos cambia (ver más arriba).
2. Sobre hechos concretos y verificables (citas legales, códigos de
   repuesto, medidas técnicas) no se observó ninguna diferencia de exactitud
   entre modelos en este corpus — las diferencias están en la estructura del
   grafo (cuántas relaciones no-hub) y en la granularidad (cuántas entidades
   por chunk), no en si los datos extraídos son correctos.
3. Los 4 modelos manejan igual de bien las páginas en blanco (sin
   alucinaciones) — un buen mínimo común denominador antes de decidir cuál
   usar en producción.
4. Paralelizar por documento, no solo por chunk, es una palanca de latencia
   real y gratuita cuando hay varios documentos que indexar: ~3× menos
   tiempo de pared en este lote frente a procesarlos en serie.
5. El límite de contexto de un modelo candidato (aquí, los 16.384 de
   Mistral-Small) puede acabar limitando el `--max-tokens` que se le puede
   dar a *todos* los modelos de la comparación si se corren juntos — al
   evaluar un modelo nuevo para producción, conviene conocer su ventana de
   contexto antes de fijar los límites de generación del silo.
6. **Qwen3-4B-Instruct es la alternativa más recomendable para el rol
   `keyword` de las cuatro probadas**: único modelo sin ningún defecto en
   las 7 preguntas (ni la alucinación puntual de Qwen3-30B ni la
   verbosidad/redundancia sistemática de gpt-5.4-nano en 3 de 7), con
   respuestas idénticas a las de Qwen3-30B en varias preguntas — el modelo
   30B no aporta calidad adicional en esta tarea concreta, solo más coste
   de VRAM. Con solo 4B parámetros cabe holgadamente junto a Qwen3-30B
   (extract) en la misma GPU, a diferencia de Mistral-Small (24B), que hubo
   que retirar por falta de VRAM — la razón práctica, además de la de
   calidad, para preferirlo como sustituto de Mistral en `LIGHTRAG_KEYWORD_MODEL`.
7. El tamaño del modelo no predice la calidad en el rol `keyword`: el
   modelo más grande de los cuatro (Qwen3-30B, 30B) tuvo el único fallo de
   alucinación; el más pequeño (Qwen3-4B) tuvo cero fallos. Para una tarea
   tan acotada (dos listas cortas de un esquema fijo, sobre una consulta de
   una frase), un modelo pequeño bien instruido puede igualar o superar a
   uno mucho mayor — no extrapolar esta conclusión al rol `extract`, donde
   el volumen de contenido a procesar por chunk es muy distinto.

## Reproducir

```bash
python scripts/compare_extraction.py --self-check   # valida el parser sin red

# Rol extract — un documento por carpeta de salida, los 4 en paralelo:
for f in CDOC000810 CDOC001004 CDOC001009 CDOC001048; do
  docker compose exec -T backend python /app/scripts/compare_extraction.py \
    /app/data/${f}.pdf \
    -m openai:gpt-5.4-mini \
    -m openai:gpt-5.4-nano \
    -m "openai:Qwen3-30B-A3B-Instruct@http://<qwen-vllm-host>:8095/v1" \
    -m "openai:Mistral-Small-3.2-24B-Instruct@http://<mistral-vllm-host>:8096/v1" \
    --json --max-chunks 999 --max-tokens 10000 --out "/app/graph-bench/${f}" &
done
wait

# Rol keyword — un fichero con una pregunta por línea (ver tabla arriba).
# Mistral-Small ya no está desplegado (no cabía en VRAM junto al resto);
# se sustituyó por Qwen3-4B en LIGHTRAG_KEYWORD_MODEL:
docker compose exec -T backend python /app/scripts/compare_extraction.py \
  /app/ruta/queries.txt --role keyword \
  -m openai:gpt-5.4-nano \
  -m "openai:Qwen3-30B-A3B-Instruct@http://<qwen-vllm-host>:8095/v1" \
  -m "openai:Qwen3-4B-Instruct@http://<qwen4b-vllm-host>:8097/v1" \
  --out /app/keyword-bench
```

Salida por documento: `graph-bench/<doc>/summary.csv` (una fila por
modelo×chunk) y `graph-bench/<doc>/raw.jsonl` (entidades/relaciones
completas + texto crudo del modelo). Los 4 PDF fuente están en
[`data/`](../../data/) en la raíz del repo.
