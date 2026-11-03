# LightRAG (`lightrag-hku`) en Mattin AI

Cómo se instala, configura y usa LightRAG en este repositorio. Cada afirmación
incluye el archivo y la función donde comprobarla.

> **Ámbito**: LightRAG es una integración **opcional y opt-in**. Solo se activa
> cuando `LIGHTRAG_ENABLED=true` **y** un silo concreto tiene
> `vector_db_type == 'LIGHTRAG'`. Con `LIGHTRAG_ENABLED=false` (el valor por
> defecto) el backend arranca sin Neo4j ni `lightrag-hku`.

---

## 1. Versión exacta

| Dónde | Versión | Comprobar en |
|---|---|---|
| **Instalada en el contenedor en ejecución** | **`1.5.6`** | `docker compose run --rm backend python -c "import importlib.metadata as m; print(m.version('lightrag-hku'))"` → `1.5.6` |
| **Fijada por el Dockerfile** | `1.5.6` | `backend/Dockerfile:75` — `pip install ... "lightrag-hku[offline-storage]==1.5.6"` |
| **Referida por los comentarios del código** | `1.5.6` | `backend/tools/vector_stores/lightrag_store.py`, `backend/tools/vector_stores/lightrag/adapters.py` |
| **Comentario en pyproject** | `1.5.6` | `pyproject.toml:174` |

> ✅ **Versión alineada**: el pin del `Dockerfile`, `backend/requirements.txt` y
> los comentarios del código dicen todos `1.5.6`.
>
> **Convención de referencias**: fichero + nombre de símbolo, **sin número de
> línea**. Los números caducan en cada release de LightRAG y en cada commit
> nuestro: en el bump `1.5.5rc1` → `1.5.6`, de 8 símbolos comprobados 6 se
> movieron y uno 640 líneas (`apipeline_process_enqueue_documents`, 943 → 1585).
> Localiza con `grep -n "def <símbolo>"`.

**Cómo se instala** (no vía Poetry): `lightrag-hku` se instala **manualmente
con pip en el Dockerfile**, no como dependencia de `pyproject.toml`, para
esquivar un conflicto de `pgvector` (`langchain-postgres` pide `<0.4`,
`lightrag-hku` pide `>=0.4.2`). El Dockerfile fuerza primero `pgvector>=0.4.2`
con `--force-reinstall --no-deps` y luego instala LightRAG. Comprobar en
`backend/Dockerfile:69-75` y el comentario en `pyproject.toml:172-174`.

---

## 2. Servicio Docker que contiene LightRAG

**Servicio `backend`** (contenedor `mattin-backend`). Comprobar en:

- `docker/docker-compose.yaml:42` — servicio `backend`, `dockerfile: backend/Dockerfile`.
- Variables de entorno relevantes en `docker/docker-compose.yaml:88-91`:
  ```yaml
  LIGHTRAG_ENABLED: ${LIGHTRAG_ENABLED:-true}
  NEO4J_URI: ${NEO4J_URI:-bolt://neo4j:7687}
  NEO4J_USERNAME: ${NEO4J_USERNAME:-neo4j}
  NEO4J_PASSWORD: ${NEO4J_PASSWORD:-neo4j}
  ```
- El servicio `neo4j` (imagen `neo4j:5-community`, plugin APOC) está en
  `docker/docker-compose.yaml:171-180`; `backend` depende de él
  (`depends_on: neo4j`, línea 100-101).

LightRAG usa **tres backends de almacenamiento**, todos servicios del compose:
`neo4j` (grafo), `qdrant` (vectores, por defecto) o `postgres` (vectores
alternativo + KV + doc-status). Ver §8.

---

## 3. Configuración

### 3.1 Configuración global (env / `backend/config.py`)

| Variable | Default | Comprobar en |
|---|---|---|
| `LIGHTRAG_ENABLED` | `false` | `backend/config.py` |
| `NEO4J_URI` | `None` | `backend/config.py` |
| `NEO4J_USERNAME` | `neo4j` | `backend/config.py` |
| `NEO4J_PASSWORD` | `None` | `backend/config.py` |
| `ENTITY_EXTRACT_MAX_GLEANING` | `0` | `backend/config.py` |
| `LIGHTRAG_EXTRACT_MAX_TOKENS` | sin tope | tope de salida del rol `extract` (ver 9.1.3) |
| `LIGHTRAG_EXTRACT_GUIDED_JSON` | `true` | manda el esquema como `response_format` (ver 9.1.3) |
| `MAX_PARALLEL_INSERT` | `3` | documentos (= páginas) en paralelo; leída por LightRAG al importar |
| `MAX_ASYNC_LLM` | `4` | llamadas LLM concurrentes por rol; leída por LightRAG al importar |
| `EMBEDDING_FUNC_MAX_ASYNC` | `8` | llamadas de embedding concurrentes |
| `SQLALCHEMY_DATABASE_URI` | — | usado como `config.DATABASE_URL` para PG KV/doc-status (`storage_config.py`) |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` | `storage_config.py` |
| `VECTOR_DB_TYPE` | `PGVECTOR` | usado por el silo, no por LightRAG directamente |

> Nota: el default del repo para el *gleaning* es `0` (`config.py`), mientras
> que el default nativo de LightRAG `1.5.6` es `DEFAULT_MAX_GLEANING = 1`
> (`lightrag/constants.py`, campo `entity_extract_max_gleaning` en
> `lightrag/lightrag.py`).

### 3.2 Configuración por-silo (columnas en la tabla `Silo`)

Cada silo LightRAG se configura de forma independiente. Comprobar en
`backend/models/silo.py`:

| Columna | Tipo | Rol |
|---|---|---|
| `vector_db_type` | String | `'LIGHTRAG'` activa este backend (`silo.py`) |
| `lightrag_vector_db_type` | String, def. `'QDRANT'` | qué vector store usa LightRAG por dentro: `QDRANT` o `PGVECTOR` (`silo.py`) |
| `lightrag_chunk_strategy` | String | `fixed_token`/`recursive_character`/`semantic_vector`/`paragraph_semantic` (`silo.py`) |
| `lightrag_chunk_token_size` | Integer | tamaño de chunk (`silo.py`) |
| `lightrag_chunk_overlap_token_size` | Integer | solape (`silo.py`) |
| `lightrag_language` | String | idioma de extracción de entidades/keywords → `addon_params['language']` (`silo.py`) |
| `lightrag_entity_extract_max_gleaning` | Integer | iteraciones de *gleaning* (`silo.py`) |
| `lightrag_max_source_ids_per_entity` | Integer | tope de source-ids por entidad (`silo.py`) |
| `lightrag_max_source_ids_per_relation` | Integer | tope de source-ids por relación (`silo.py`) |
| `lightrag_entity_types` | Text | categorías de entidad, separadas por comas → se traduce a `addon_params['entity_types_guidance']` (`silo.py`, ver §6) |
| `lightrag_graph_context_enabled` | Boolean, def. `false` | expone el grafo como burbuja en el playground (`silo.py`) |
| `use_agent_as_query` | Boolean, def. `false` | usa el LLM del agente para sintetizar (`silo.py`) |

**LLMs por rol** (una `AIService` independiente por rol — `silo.py`):
`extract_service_id`, `keywords_service_id`, `query_service_id`,
`vlm_service_id`. `indexing_service_id` es un alias legacy que mapea a
`extract`. La validación de que un silo LightRAG requiere al menos un servicio
`extract` + `embedding_service` está en `SiloService.create_or_update_silo`
(`backend/services/silo_service.py`).

**Modo de query por-agente** (única perilla LightRAG específica del agente):
`Agent.lightrag_query_mode` (`backend/models/agent.py`), valores
`skill-routed | local | global | hybrid | mix | naive | bypass`, `NULL` para
silos no-LightRAG.

---

## 4. Clases y funciones principales

### 4.1 En este repositorio (capa de integración)

Todo el código LightRAG-específico vive en
`backend/tools/vector_stores/lightrag/` + `lightrag_store.py`, con **imports
perezosos** de `lightrag` dentro de cada método, para que el backend importe
sin la dependencia instalada.

| Símbolo | Archivo | Rol |
|---|---|---|
| `LightRAGStore` | `lightrag_store.py` | Implementa `VectorStoreInterface`; un silo = un *workspace* `silo_{id}` |
| `LightRAGStore._build_rag` | `lightrag_store.py` | Construye la instancia `LightRAG(...)` (sin inicializar) |
| `LightRAGStore._get_rag_instance` | `lightrag_store.py` | Cache síncrona (indexación / hilos de fondo) |
| `LightRAGStore._aget_rag_instance` | `lightrag_store.py` | Cache async con lock por-colección; inicializa Neo4j en el loop del caller |
| `LightRAGRetriever` | `lightrag_store.py` | `BaseRetriever` de LangChain sobre `aquery_llm` |
| `build_llm_model_func` | `lightrag/adapters.py` | Envuelve una `AIService` en la firma que espera LightRAG |
| `build_role_llm_configs` | `lightrag/adapters.py` | `dict[str, RoleLLMConfig]` para roles `extract/keyword/vlm` |
| `build_embedding_func` | `lightrag/adapters.py` | Construye `EmbeddingFunc` desde una `EmbeddingService` |
| `is_lightrag_available` | `lightrag/adapters.py` | Feature flag: `LIGHTRAG_ENABLED` + `import lightrag` OK |
| `build_storage_config` | `lightrag/storage_config.py` | Elige backends + exporta env vars (`NEO4J_*`, `QDRANT_*`, `POSTGRES_*`) |
| `IndexingTokenAccumulator` | `lightrag/token_accumulator.py` | Suma tokens LLM/embedding de un run de indexación |
| `SiloGraphService` | `services/silo_graph_service.py` | Lee el grafo de un silo desde Neo4j (Cypher directo) |

### 4.2 En LightRAG instalado (`1.5.6`)

Raíz: `/usr/local/lib/python3.11/site-packages/lightrag/`.

| Símbolo | Archivo | Rol |
|---|---|---|
| `LightRAG` (dataclass) | `lightrag/lightrag.py` | Clase principal; mezcla `_RoleLLMMixin`, `_StorageMigrationMixin`, `_PipelineMixin` |
| `LightRAG.initialize_storages` | `lightrag/lightrag.py` | Inicializa los objetos de storage del workspace |
| `LightRAG.ainsert` | `lightrag/lightrag.py` | API alto nivel de inserción |
| `apipeline_enqueue_documents` | `lightrag/pipeline.py` | Encola documentos en `doc_status` |
| `apipeline_process_enqueue_documents` | `lightrag/pipeline.py` | Procesa la cola (chunk + extracción) |
| `LightRAG.aquery` | `lightrag/lightrag.py` | Wrapper compat → solo `str`/stream |
| `LightRAG.aquery_llm` | `lightrag/lightrag.py` | Devuelve dict con contexto + grafo (lo que usa el repo) |
| `LightRAG.aquery_data` | `lightrag/lightrag.py` | Solo recuperación (`only_need_context=True`) |
| `QueryParam` (dataclass) | `lightrag/base.py` | Parámetros de query |
| `extract_entities` | `lightrag/operate.py` | Extracción de entidades/relaciones |
| `kg_query` / `naive_query` | `lightrag/operate.py` | Motores de query grafo / vectorial |
| `RoleLLMConfig` | `lightrag/llm_roles.py` | Config LLM por rol |
| `resolve_chunk_options` | `lightrag/parser/routing.py` | Resuelve opciones de chunking |
| `EmbeddingFunc` (dataclass) | `lightrag/utils.py` | Envoltorio de embeddings |

**Construcción de la instancia** (`LightRAGStore._build_rag`,
`lightrag_store.py`): el repo pasa
`working_dir` (tempdir), `workspace=silo_{id}`, `llm_model_func` (base/fallback),
`role_llm_configs`, `embedding_func`, los cuatro backends de storage,
`entity_extract_max_gleaning` y `**extra_kwargs` (chunk sizes + source-id caps
solo si están configurados). El `language` se inyecta mutando
`rag.addon_params['language']` **después** de construir, para no pisar los
defaults de LightRAG (final de `_build_rag`).

---

## 5. Pipeline de inserción

**Punto de entrada del repo**: `LightRAGStore.index_documents`
(`lightrag_store.py`). Flujo:

1. Filtra docs vacíos; alinea `texts[]` con `file_paths[]` (etiqueta de origen
   por chunk vía `_source_label_from_metadata`, `lightrag_store.py`).
2. Activa un `IndexingTokenAccumulator` en un `contextvar`
   (`set_active_accumulator`, `adapters.py`) para contar tokens.
3. Llama a `_ainsert_with_progress` → `_ainsert` (`lightrag_store.py`).
4. Devuelve un dict de métricas (`totals()` de `token_accumulator.py`) que
   se persiste en la tabla `indexing_metric` (`models/indexing_metric.py`).

**`_ainsert`** (`lightrag_store.py`) **no** llama al `rag.ainsert` normal;
en su lugar usa la ruta de chunking "moderna" explícitamente:

```python
chunk_opts = resolve_chunk_options(rag.addon_params, process_options=process_options)
await rag.apipeline_enqueue_documents(texts, file_paths=..., process_options="F", chunk_options=chunk_opts)
await rag.apipeline_process_enqueue_documents()
```

Motivo (docstring en `lightrag_store.py`): pasar `process_options` evita
que LightRAG caiga en la ruta legacy de 6 args (`Chunking F(legacy)`) y usa el
contrato nuevo del chunker (`Chunking F`).

**Selector de estrategia de chunking** (`lightrag_store.py`,
`_CHUNK_STRATEGY_OPTION`): mapea la estrategia del silo al char de
`process_options` de LightRAG:

| Estrategia del silo | `process_options` | Constante LightRAG |
|---|---|---|
| `fixed_token` | `"F"` | `PROCESS_OPTION_CHUNK_FIXED` |
| `recursive_character` | `"R"` | `PROCESS_OPTION_CHUNK_RECURSIVE` |
| `semantic_vector` | `"V"` | `PROCESS_OPTION_CHUNK_VECTOR` |
| `paragraph_semantic` | `"P"` | `PROCESS_OPTION_CHUNK_PARAGRAH` |

Desconocido → `"F"`. Comprobar constantes en `lightrag/constants.py`.

**Dentro de LightRAG** (`1.5.6`):
- `apipeline_enqueue_documents` (`pipeline.py`): genera IDs MD5 + dedup por
  contenido, crea el estado inicial en `doc_status`, filtra ya-procesados.
  `chunk_options` (cuando es `None`) se deriva de `addon_params['chunker']` vía
  `resolve_chunk_options` y se persiste en `full_docs[doc_id]['chunk_options']`.
- `apipeline_process_enqueue_documents` (`pipeline.py`): guard de un-solo-
  worker vía `pipeline_status["busy"]` bajo `pipeline_status_lock` (por-workspace,
  `kg/shared_storage.py`); trocea, ejecuta extracción de entidades por chunk,
  actualiza estado. Los documentos se procesan de 3 en 3
  (`max_parallel_insert`, env `MAX_PARALLEL_INSERT`), y dentro de cada uno los
  chunks de 4 en 4 (`llm_model_max_async`, env `MAX_ASYNC_LLM`).
- **Progreso**: `_ainsert_with_progress` (`lightrag_store.py`) poll-ea cada
  0.5s `rag.doc_status.get_status_counts()` y cuenta `processed + failed` menos
  un baseline tomado antes de empezar (los documentos de recursos anteriores del
  mismo silo ya están `processed`). **No** usa `pipeline_status["cur_batch"]`:
  ese contador se incrementa cuando un documento *entra* al semáforo
  (`pipeline.py`), no cuando termina, así que las primeras
  `max_parallel_insert` páginas parecían instantáneas y el ETA salía sesgado a
  la baja. `pipeline_status` no tiene ningún contador de completados.
- `resolve_chunk_options` (`parser/routing.py`) lee `addon_params['chunker']`,
  recorta al sub-dict de la estrategia seleccionada por `process_options` y
  devuelve una copia profunda independiente.

**Borrado**: `delete_documents` y `delete_documents_excluding` son **no-ops
deliberados** (`lightrag_store.py`, marcados `ponytail:`) — LightRAG no
soporta borrado per-documento del grafo. Reindexar es idempotente (doc_id =
hash de contenido); los chunks huérfanos son el techo conocido. El borrado real
es a nivel de silo: `delete_collection` (`lightrag_store.py`) limpia Neo4j,
Qdrant y PostgreSQL directamente (helpers `_cleanup_neo4j/_cleanup_qdrant/
_cleanup_postgres`, `lightrag_store.py`).

### 5.1 Rodaje del feeder (ventana deslizante)

> **Contexto histórico**: el análisis y las cifras de esta sección (154
> páginas, 15 minutos, s/chunk por ventana) se midieron el 2026-08-27, bajo el
> esquema antiguo de un documento LightRAG por página. Desde el refactor de
> documento-por-fichero (§13), un recurso es un único documento LightRAG, no
> N páginas — el mecanismo de ventana deslizante en sí no cambia, pero
> "página" en esta sección significa "documento LightRAG bajo el esquema
> antiguo", no la unidad actual. Ver "Ajuste tras el refactor" al final de
> esta sección para la guía de tuning vigente.

`INGESTION_FEED_WINDOW` (`resource_service.py`, `_feed`) mantiene N
**recursos** en vuelo a la vez, no N páginas. Un recurso no libera su hueco
hasta estar **terminado entero** — aunque tenga 90 páginas y
`MAX_PARALLEL_INSERT` tenga huecos libres de sobra. Con window pequeño y los
primeros recursos grandes, el siguiente recurso puede tardar minutos en entrar
aunque haya cientos de páginas listas para procesarse en paralelo.

Observado en producción (silo Domusa, 2026-08-27, `MAX_PARALLEL_INSERT=48`,
`window=3`): los 3 primeros recursos (91+19+44=154 páginas) tardaron **15
minutos** en liberar hueco para el 4º. El ritmo de indexado real (chunks
completados por segundo) también se observó acelerando con el tiempo — 4.66 →
1.84 → 1.42 s/chunk en ventanas sucesivas de la misma tanda — consistente con
un rodaje de concurrencia al principio del lote, no con que el trabajo se
volviera más ligero.

**Ojo con el razonamiento previo sobre este valor**: el comentario original en
`.env`/`.env.example` asumía que "3 ficheros de tamaño medio ya producen más
páginas en vuelo que huecos en `MAX_PARALLEL_INSERT`, así que subir la ventana
no da más rendimiento". Con 154 páginas para un tope de 48, esa condición se
cumplía numéricamente — y aun así hubo 15 minutos de espera. Eso apunta a que
el cuello de botella real no es la disponibilidad de páginas, sino algo más
fino en el ritmo de despacho/extracción al arrancar (posiblemente el trabajo
de extracción de texto por página, que corre en un thread pool y es más
secuencial que las llamadas LLM). **No verificado a fondo** — perfilar el
arranque (cuánto tarda cada etapa: extracción, chunking, primera llamada LLM)
sería el siguiente paso si se quiere una respuesta firme.

`INGESTION_FEED_WINDOW` se subió de 3 a 8 el 2026-08-27 como experimento, sin
certeza de que resuelva el rodaje — si acorta la fase lenta inicial en la
siguiente indexación, mantenerlo; si no, la causa está en otro sitio y subir
la ventana no es la palanca correcta. Tratar este valor como algo a ajustar
por despliegue, no como una constante ya cerrada.

**Ajuste tras el refactor de documento-por-fichero.** Como un recurso es ahora
UN documento LightRAG, el paralelismo real a nivel de documento es
`min(INGESTION_FEED_WINDOW, MAX_PARALLEL_INSERT)` — con los defaults de
`.env.example` (`MAX_PARALLEL_INSERT=48`, `INGESTION_FEED_WINDOW=8`), solo 8
de los 48 huecos de `MAX_PARALLEL_INSERT` se usan nunca, salvo que se suba la
ventana. La afirmación anterior de que subir la ventana "no da más
rendimiento" **ya no es cierta en general**: para un corpus de muchos
ficheros PEQUEÑOS, cada fichero solo ocupa unos pocos de sus `MAX_ASYNC_LLM`
huecos de extracción (pocos chunks por fichero), así que más ficheros en
vuelo SÍ añade paralelismo real; para un corpus de pocos ficheros GRANDES, el
cuello de botella sigue siendo `MAX_ASYNC_LLM` dentro de cada fichero, no la
ventana.

Regla práctica:
- **Muchos ficheros pequeños**: subir `INGESTION_FEED_WINDOW` hacia,
  aproximadamente, `MAX_ASYNC_LLM ÷ chunks medios por fichero`, y mantener
  `MAX_PARALLEL_INSERT >= INGESTION_FEED_WINDOW` (todo lo que exceda la
  ventana queda sin usar).
- **Pocos ficheros grandes**: la palanca real es `MAX_ASYNC_LLM`, no la
  ventana.
- En ambos casos, la señal práctica de ajuste sigue siendo la misma de
  siempre: mirar el log del servidor de inferencia (`Waiting: 0`, uso de KV
  cache de la GPU) y subir mientras haya margen — ver el comentario de
  `MAX_PARALLEL_INSERT` en `.env.example`.

### 5.2 Filas `failed` en `doc_status` que no son fallos

Un `SELECT status, count(*) FROM lightrag_doc_status` suele dar cientos de filas
`failed` en un silo sano. Antes de concluir que falta contenido, mirar el
`error_msg`: casi todas son rechazos de dedup que LightRAG deja escritos como
filas propias, con id `dup-<hash>`, mientras el documento original está
`processed`.

> **Desde el esquema por-fichero (§13): la unidad del dedup es ahora el
> FICHERO, no la página.** El mecanismo SQL de abajo no cambia — sigue siendo
> `apipeline_enqueue_documents` comparando contenido/nombre — pero antes un
> texto repetido (p. ej. una página de boilerplate idéntica en varios
> manuales) generaba una fila `dup-*` por CADA página repetida; ahora, al ser
> cada fichero un único documento, genera como mucho una fila `dup-*` por
> fichero entero. Las observaciones concretas de esta sección (2026-10-01) se
> tomaron bajo el esquema antiguo, un documento LightRAG por página — quedan
> como referencia histórica de cómo leer `error_msg`, no como cifras vigentes.

| `error_msg` | Qué es | ¿Falta contenido? |
|---|---|---|
| `File name already exists. Original doc_id: …, Status: processed` | Reindexar un recurso vuelve a encolar documentos que ya estaban. Cada reindexado deja una fila más (antes, una por página; ahora, una por fichero). | No |
| `Identical content already exists under another filename. Original doc_id: …` | El documento (antes, una página; ahora, el fichero entero) es idéntico, palabra por palabra, a otro **ya indexado**. LightRAG lo indexa una sola vez, atribuido al original. | Para el documento duplicado, sí |

Consulta para separar los fallos reales:

```sql
SELECT count(*) FROM lightrag_doc_status
WHERE workspace = 'silo_<id>' AND status = 'failed' AND id NOT LIKE 'dup-%';
```

Si da 0, nada falló de verdad.

**El segundo caso afecta a la recuperación.** Los chunks del documento
duplicado solo existen con el `file_path` del original. Por eso:

- `list_documents_mentioning` (la herramienta de cobertura) nunca devuelve el
  documento duplicado para un término que solo aparece en él.
- Una pregunta cuyo `docs_esperados` incluye el documento duplicado pierde
  `doc_recall` sin que el agente lo haya hecho mal.

Observado el 2026-10-01 en los silos EN de DOMUSA (`silo_39`
`test_new_embedding_en` y `silo_47` `test_new_dedup_en`), bajo el esquema
por-página anterior a este refactor:

- **Fallos reales:** 0 en los dos silos. Las 33 fuentes tienen páginas procesadas en ambos.
- **Rechazos por nombre:** 545 en `silo_39` y 592 en `silo_47` (más reindexados).
- **Contenido idéntico entre manuales:**

  | Páginas duplicadas | Original indexado |
  |---|---|
  | CDOC004388, p. 1, 2, 4 y 6 | CDOC004020 |
  | CDOC001134, p. 33 | CDOC001110 |

  12 de las 63 preguntas del eval set GB (`eval_set_gb_261001.json`) esperan
  alguno de estos documentos. Al puntuarlas, comprobar si el dato está en una
  página duplicada antes de contar el documento ausente como fallo.

Posible mejora, sin hacer: añadir al `file_path` algo propio del recurso para
que dos manuales con contenido igual no colisionen. A cambio se pierde el ahorro
de no indexar dos veces el mismo texto.

---

## 6. Extracción de entidades

**Función LightRAG**: `extract_entities` (`operate.py`).

- Usa el LLM del rol **`extract`** (`global_config["role_llm_funcs"]["extract"]`,
  `operate.py`) — en el repo, el que construye `build_role_llm_configs`
  desde `silo.extract_service` (`adapters.py`, `lightrag_store.py`).
- **Gleaning**: `entity_extract_max_gleaning` (`operate.py`). En
  `1.5.6` corre **como mucho una vez** (`run_gleaning = ... > 0`,
  `operate.py`); reejecuta el par user/assistant inicial con un prompt
  "continue". Se salta si el payload combinado excede `MAX_EXTRACT_INPUT_TOKENS`.
  El valor viene del silo (`lightrag_entity_extract_max_gleaning`) o del default
  global `config.ENTITY_EXTRACT_MAX_GLEANING` (`lightrag_store.py`).
- **Idioma**: `addon_params['language']` (`operate.py`), inyectado por
  el repo desde `silo.lightrag_language` (`lightrag_store.py`).
- **Tipos de entidad**: se resuelven en un *prompt profile* vía
  `resolve_entity_extraction_prompt_profile(addon_params, ...)` (en `prompt.py`);
  el `entity_types_guidance` sale de `addon_params['entity_types_guidance']` si
  es un string no vacío, si no del default `PROMPTS["default_entity_types_guidance"]`.
  **LightRAG no acepta una lista plana** (`addon_params['entity_types']` no lo lee
  nada del pipeline) — solo el string de `entity_types_guidance`.
  El repo sobreescribe esto desde `silo.lightrag_entity_types` (texto separado por
  comas en "Advanced settings") vía `_build_entity_types_guidance()`
  (`lightrag_store.py`), que genera el mismo formato (frase + bullets, sin
  descripciones por tipo) que usa el default de LightRAG. Si el campo está vacío:
  con `lightrag_language == 'Spanish'` se inyecta una traducción propia de los 11
  tipos por defecto de LightRAG (`_SPANISH_DEFAULT_ENTITY_TYPES`, ver §3, LightRAG
  no localiza esto por sí mismo); en cualquier otro caso no se toca `addon_params`
  y se usa el default nativo en inglés.
- **Prompts** (`lightrag/prompt.py`, dict `PROMPTS`): delimitadores
  `PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"` y
  `PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"`; claves
  `entity_extraction_system_prompt`, `entity_extraction_user_prompt`,
  `keywords_extraction`.
  Hay variantes JSON (`entity_extraction_json_*`) según
  `global_config["entity_extraction_use_json"]`.

**Grafo resultante**: cada nodo/relación se guarda en Neo4j con el nombre del
workspace (`silo_{id}`) como **label** del nodo (no como propiedad) — clave para
entender `SiloGraphService` (§9). Comprobar en `silo_graph_service.py` y
el helper de limpieza `_cleanup_neo4j` (`lightrag_store.py`,
``MATCH (n:`silo_X`) DETACH DELETE n``).

### 6.1 Deduplicado/merge de entidades

LightRAG fusiona entidades por **coincidencia exacta de `entity_name`** tras
normalizarlo con `normalize_entity_name()` → `normalize_extracted_info()`
(`utils.py`): limpieza de tags HTML, caracteres chinos full-width → ASCII,
comillas envolventes, NBSP, y filtrado de nombres puramente numéricos cortos
(`< 3` dígitos, o patrones `x.y.z` de `< 6` caracteres). Sin la capa del repo
(descrita abajo) **no hay lowercase, fuzzy matching, embeddings ni ninguna
resolución de entidades real** — dos nombres que difieren en mayúsculas,
tildes, o forma (acrónimo vs. nombre completo) quedan como nodos separados. El
merge en sí ocurre en `_merge_nodes_then_upsert()` (`operate.py`): funde
`source_id`s, vota el `entity_type` mayoritario, dedupea descripciones exactas
y, si hay varias descripciones distintas, dispara un resumen vía LLM
(`_handle_entity_relation_summary`) — pero solo entre nodos que ya coincidían
carácter a carácter.

**Resolución de variantes de spelling — IMPLEMENTADO (2026-09-30).** El repo
pliega el nombre a una **clave canónica de fusión** en
`tools/vector_stores/lightrag/entity_name_normalization.py`:

- `canonicalize_entity_name()`: NFKC → NFKD + strip de marcas de acento →
  NFC → casefold → guiones separadores (`-`, en/em dash, soft hyphen) →
  espacio → colapso de espacios. Determinista, sin estado. Así
  `MCF-40`, `mcf 40` y `Mcf 40` llegan a LightRAG con el MISMO
  `entity_name` y su `_merge_nodes_then_upsert()` nativo los funde — no se
  parchea el merge de la librería, solo el punto donde se calcula el nombre.
- El patch (idempotente, instalado en `LightRAGStore._build_rag`) reescribe
  `lightrag.utils.normalize_entity_name` **y** `lightrag.operate.
  normalize_entity_name` (operate lo importa por nombre; rebindear solo
  utils no basta). Conserva el contrato de limpieza de LightRAG: nombres
  puramente numéricos cortos siguen devolviendo `""` y el caller los
  descarta igual que antes.
- `MCF40` (sin separador) **no** se funde con `MCF-40`: el plegado unifica
  variantes del mismo nombre escrito distinto, no formas compactas.
- **Display name = variante más frecuente.** El patch cuenta cada llamada
  como una *mención* (un registro de entidad + un endpoint de relación
  cuentan por separado — sesgo aceptable: ambas formas salen de la misma
  extracción) en un `EntityNameVariantCollector` por corrida, con el mismo
  patrón contextvar que el acumulador de tokens. Al terminar la corrida
  (`index_documents` y `process_enqueued_documents`) el wrapper drena el
  collector y persiste en la tabla `entity_name_variant`
  (`silo_id, canonical_name, variant, mention_count`, upsert que SUMA al
  conflicto, silo con CASCADE). Persistencia best-effort: si falla, la
  indexación sigue y la próxima corrida re-acumula.
- En el render del grafo (`aretrieve_graph_context` →
  `_normalize_lightrag_graph(raw_data, name_map)`) cada nombre canónico se
  muestra como su variante más mencionada (empate: primera vista; sin
  filas: se muestra la clave canónica tal cual). Solo presentación — la
  identidad del nodo sigue siendo la clave canónica.
- **La búsqueda del explorador de grafo también pliega el término**
  (`SiloGraphService._build_search_clause`): además de las condiciones
  originales (spelling crudo, para descripciones —que son prosa con
  tildes— y nodos legacy), añade `n.entity_id CONTAINS $canonical_search`
  con el término plegado, así "MCF-40" llega al nodo `mcf 40`. El match de
  descripciones queda tal cual (los textos sí llevan tildes). Y el
  explorador reescribe ids/aristas con el mismo mapa de display que la
  burbuja (`get_silo_graph(..., db=...)`) — sin eso, tras reindexar el
  explorador mostraría claves canónicas mientras la burbuja muestra
  variantes. Aprovechando la revisión del fichero se cerró además una
  **inyección Cypher pre-existente**: `node_label` se interpolaba sin
  validar y ahora pasa por whitelist (`^[A-Za-z0-9_\- ]{1,64}$`,
  ValueError → 422).
- **La burbuja de grafo del playground también usa el display**: el
  retriever aplica el mapa al `lightrag_raw_data` que viaja al frontend
  (`_wrap_query_response(..., name_map=...)`). Importante porque
  `merge_lightrag_graph` (streaming) deduplica entidades **por id**, y sin
  el mapa la burbuja mostraría las claves canónicas del LLM-context en vez
  de las variantes que muestra el explorador. El string de contexto que ve
  el LLM **no** se reescribe: sigue reflejando lo que LightRAG indexó
  (nombres canónicos — cambio cosmético asumido en los prompts).
- El patch rebinds cualquier módulo `lightrag.*` ya importado cuyo
  atributo apunte al original (escaneo de `sys.modules`), no solo
  `operate.py` — sobrevive a upgrades que muevan el punto de importación;
  verificar igualmente al subir de versión. Instalación con lock
  (`_INSTALL_LOCK`) y `utils` rebindeado primero, así una importación que
  ocurra durante el patch ya recibe el wrapper.
- La clave canónica se recorta al límite de nombre de LightRAG
  (`DEFAULT_ENTITY_NAME_MAX_LENGTH` = 256 chars — la librería trunca tras
  normalizar), para que las filas de variantes apunten al nombre real del
  nodo, no a una versión más larga que jamás existirá.
- El display **sigue la frecuencia, sin pinado**: si una variante supera
  estrictamente a la vigente, el nombre visible cambia (empate: primera
  vista). El churn cosmético entre turnos se asume; pinar al incumbent
  exigiría estado adicional sin beneficio claro.
- **La tabla `entity_name_variant` no lleva índices extra**: la constraint
  única `(silo_id, canonical_name, variant)` ya cubre el upsert, el scan de
  cascade del FK y el lookup de display (prefijo `(silo_id, canonical_name)`;
  el sort por `mention_count DESC` ordena un puñado de filas por grupo).
- El mapa de display se consulta con **cache por silo (TTL 60 s)**,
  invalidada al persistir una corrida — el lookup está en el hot path de
  query, y sin cache serían 3-4 pool checkouts por turno de agente.
- **Edge cases asumidos**: los conteos son menciones (registro de entidad +
  endpoints de relación); `MCF40` compacto ≠ `MCF-40`; dos nombres > 500
  chars con el mismo prefijo plegado colisionarían (impráctico — el write
  trunca); corridas concurrentes sobre el mismo silo componen SIN perder
  conteos (el `ON CONFLICT` suma atómicamente; ojo: la serialización del
  pipeline por silo es otro asunto pre-existente — varios paths de ingesta
  no pasan por `silo_indexing_lock`); un crash a mitad de corrida pierde
  las menciones de esa corrida (cosmético); grafos legacy conservan nombres
  bonitos hasta reindex/limpieza retroactiva.
- **Limitaciones**: grafo indexado ANTES de este cambio conserva los
  nombres no canónicos (los duplicados legacy siguen ahí hasta reindex o
  limpieza retroactiva vía Cypher — mejora futura). Los contadores se
  pierden si el proceso muere a mitad de corrida (cosmético).

**Pendiente (subsunción, NO es canónicación)**: `mcf 40` ⊂ `quemador mcf
40` requiere clasificación variant/hyponym/noise — ver
`scripts/analyze_entity_fragmentation.py` y recomendación 5 del benchmark de
JSON/tope de tokens. Si la muestra sale mayormente *noise*, el arreglo es en
la extracción, no en el grafo.

**Mejora de implementación pendiente (no urgente)**: el filtro numérico
nativo (`< 3` dígitos puros) es limitado y deja pasar valores técnicos con
unidad como entidades sueltas — confirmado en el grafo real del silo Domusa
(`silo_11`, Neo4j): ~81 de 8488 nodos (~1%) son ruido tipo `230 V`, `45 ºC`,
`700mm`, `7 bar`, `4.500 W`, `35Hz`, `1,4 Kg`, o rangos como `30 - 85 ºC`
—probablemente atributos (presión/temperatura/caudal) mal promovidos a nodo
propio en vez de quedar como propiedad de la entidad real. Impacto detectado
bajo (nodos sin relaciones ricas, no colisionan entre sí ni con entidades
reales), por lo que se deja documentado como mejora futura y no como fix
inmediato. Si se aborda: post-filtro regex tipo
`^\d+[.,]?\d*\s*(V|A|ºC|mm|bar|W|Hz|rpm|Kg|L|Pa|Ω|cm|m|kW|mbar)$` aplicado
tras `normalize_entity_name` en `lightrag_store.py`, o limpieza retroactiva
vía Cypher sobre el grafo ya indexado. Señal para priorizarlo: si el eval de
un silo muestra respuestas citando estos valores como si fueran entidades
relevantes, o si el ratio de ruido crece con corpus más grandes.

---

## 7. Queries

**`QueryParam`** (`base.py`). Campos relevantes:
- `mode`: `local|global|hybrid|naive|mix|bypass` — **default nativo `mix`**
  (`base.py`), pero el repo usa `hybrid` por defecto.
- `only_need_context: bool` (`base.py`) — **el repo siempre lo pone `True`**.
- `top_k` (`base.py`), `chunk_top_k` (`base.py`).

**Modos válidos y mapeo del repo** (`_resolve_query_mode`, `lightrag_store.py`):
- Modos nativos (`local/global/hybrid/naive/mix/bypass`) pasan tal cual.
- `similarity` / `similarity_score_threshold` → `hybrid`; `mmr` → `mix`;
  desconocido → `hybrid`.

**Ruta de query del repo**:
1. `LightRAGRetriever` (`lightrag_store.py`) — retriever de LangChain.
   Siempre llama con `only_need_context=True` (`lightrag_store.py`), de
   modo que **LightRAG NO genera la respuesta**: devuelve solo el contexto del
   grafo y el LLM del agente sintetiza. Por eso el rol `query` se deja sin
   configurar a propósito (`adapters.py`, y el filtro de log
   `_DropUnconfiguredRoleLog` en `lightrag_store.py`).
2. Llama a `rag.aquery_llm(query, param)` (no `aquery`, que descarta el
   `raw_data`) — `lightrag_store.py`.
3. `_wrap_query_response` (`lightrag_store.py`) mete el string de contexto en
   `Document.page_content` y el grafo estructurado en
   `metadata["lightrag_raw_data"]`.
4. `_normalize_lightrag_graph` (`lightrag_store.py`) traduce la forma de
   LightRAG (`entity_name`/`src_id`/…) a la que espera el frontend
   (`id`/`name`/`source`/`target`).
5. `_extract_lightrag_keywords` (`lightrag_store.py`) limpia del contexto los
   artefactos internos: el JSON de keywords (`high_level_keywords`) y los tags
   `[Data: Reports (N)]`.

**`aquery_llm` — forma de retorno** (`lightrag.py`, verificado en
`1.5.6`):
```python
{
  "status": "success" | "failure",
  "data": {
     "entities": [{entity_name, entity_type, description, source_id, file_path, ...}],
     "relationships": [{src_id, tgt_id, description, keywords, weight, source_id, ...}],
     "chunks": [{content, file_path, chunk_id, reference_id}],
     "references": [{reference_id, file_path}],
  },
  "metadata": {"query_mode": ..., "keywords": {...}},
  "llm_response": {"content": str|None, "response_iterator": ..., "is_streaming": bool},
}
```
Como el repo usa `only_need_context=True`, el `content` es el contexto crudo, no
una respuesta generada. Semántica por modo (docstring de `aquery_data`,
`lightrag.py`): `local`=entidades+chunks, `global`=relaciones+entidades,
`hybrid`=merge round-robin, `mix`=KG+chunks vectoriales, `naive`=solo chunks
vectoriales, `bypass`=sin recuperación.

**Otra ruta** — contexto de grafo directo:
`LightRAGStore.aretrieve_graph_context` (`lightrag_store.py`) llama a
`aquery_llm` con `only_need_context=True` y devuelve el grafo normalizado
**sin descartar** cuando el contexto textual está vacío (a diferencia del
retriever). Lo usa `SiloService._search_via_lightrag_retriever`
(`silo_service.py`) para el buscador de silos.

---

## 8. Almacenamiento

Elegido en `build_storage_config` (`storage_config.py`). LightRAG
usa **cuatro** backends simultáneos:

| Rol LightRAG | Backend | Clase LightRAG | Archivo (contenedor, `1.5.6`) |
|---|---|---|---|
| `graph_storage` | **Neo4j** (siempre) | `Neo4JStorage` | `kg/neo4j_impl.py` |
| `vector_storage` | **Qdrant** (def.) o **PGVector** | `QdrantVectorDBStorage` / `PGVectorStorage` | `kg/qdrant_impl.py` / `kg/postgres_impl.py` |
| `kv_storage` | **PostgreSQL** | `PGKVStorage` | `kg/postgres_impl.py` |
| `doc_status_storage` | **PostgreSQL** | `PGDocStatusStorage` | `kg/postgres_impl.py` |

**Selección del vector store** (`storage_config.py`):
`SUPPORTED_LIGHTRAG_VECTOR_DB_TYPES = {"QDRANT": ..., "PGVECTOR": ...}`; el silo
lo elige con `lightrag_vector_db_type` (def. `QDRANT`,
`lightrag_store.py`).

**Variables de entorno exportadas** (`storage_config.py`, con
`os.environ.setdefault` para no pisar overrides del operador):
- Neo4j: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
- Postgres (parseadas del `SQLALCHEMY_DATABASE_URI` en `_parse_postgres_uri`,
  `storage_config.py`): `POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE`.
- Qdrant (solo si vector_db=QDRANT): `QDRANT_URL`, `QDRANT_API_KEY`.

**Aislamiento multi-tenant**: cada silo = un `workspace` = `silo_{id}`
(`lightrag_store.py`). En Neo4j el workspace es un **label** de nodo;
en Postgres/Qdrant es una **columna/prefijo** `workspace`. Por construcción, no
hay fuga entre silos (`silo_graph_service.py`).

**Nombres de tablas/colecciones**:
- Postgres (mapa `NAMESPACE_TABLE_MAP`, `kg/postgres_impl.py`), verificado
  en `1.5.6`:
  `LIGHTRAG_DOC_FULL`, `LIGHTRAG_DOC_CHUNKS`, `LIGHTRAG_FULL_ENTITIES`,
  `LIGHTRAG_FULL_RELATIONS`, `LIGHTRAG_ENTITY_CHUNKS`, `LIGHTRAG_RELATION_CHUNKS`,
  `LIGHTRAG_LLM_CACHE`, `LIGHTRAG_VDB_CHUNKS`, `LIGHTRAG_VDB_ENTITY`,
  `LIGHTRAG_VDB_RELATION`, `LIGHTRAG_DOC_STATUS`.
- Qdrant: colecciones con prefijo `lightrag_{workspace}_` (limpieza en
  `_cleanup_qdrant`, `lightrag_store.py`).

> 🐞 **BUG CONFIRMADO — `_cleanup_postgres` no limpia nada.** El helper
> `_cleanup_postgres` (`lightrag_store.py`) borra de tablas
> `kv_store_full_docs`, `kv_store_text_chunks`, `kv_store_full_entities`,
> `kv_store_full_relations`, `kv_store_llm_response_cache` y `doc_status`. Esos
> nombres son los del backend **JSON** de LightRAG (ficheros `kv_store_*.json`),
> **no** los del backend PostgreSQL, cuyas tablas son `LIGHTRAG_*` /
> `LIGHTRAG_DOC_STATUS` (confirmado en `NAMESPACE_TABLE_MAP`,
> `kg/postgres_impl.py`). El `DELETE` va envuelto en `try/except pass`
> (líneas 816-822), así que **falla en silencio** contra tablas inexistentes y
> retiene todas las filas del silo en Postgres al borrar el silo. El mismo
> desajuste afecta a `count_documents` (`lightrag_store.py`), que
> consulta `SELECT COUNT(*) FROM "doc_status"` → siempre devuelve 0.
> **Impacto**: al borrar un silo LightRAG, el grafo Neo4j y las colecciones
> Qdrant sí se limpian, pero las filas de `LIGHTRAG_DOC_FULL`,
> `LIGHTRAG_DOC_CHUNKS`, `LIGHTRAG_*_ENTITIES/RELATIONS`, `LIGHTRAG_LLM_CACHE` y
> `LIGHTRAG_DOC_STATUS` quedan **huérfanas para siempre**. (Nota: KV y doc-status
> viven SIEMPRE en Postgres, sea cual sea el backend vectorial, así que el bug
> aplica a todo silo LightRAG.)
> **Fix**: usar `namespace_to_table_name` de LightRAG (o mapear a los nombres
> `LIGHTRAG_*`) en `_cleanup_postgres` y `count_documents`. No verificable con
> datos en vivo hoy: aún no hay ningún silo LightRAG con tablas creadas en la DB
> (`SELECT ... WHERE tablename LIKE 'lightrag%'` devuelve vacío).

---

## 9. Modelos LLM y embeddings

### 9.1 LLM (roles)

LightRAG `1.5.6` enruta el LLM por **rol** (`llm_roles.py`): `extract`,
`keyword`, `query`, `vlm` (`ROLES` en `llm_roles.py`). Cada rol es un
`RoleLLMConfig` (`llm_roles.py`) con un `func`; **si un rol no tiene `func`,
cae al `llm_model_func` base** (`raw_func=cfg.func or base_llm_func`,
`lightrag.py`; el base no puede ser `None`, `lightrag.py`).

El repo mapea sus `AIService` a roles en `build_role_llm_configs`
(`adapters.py`):
- `extract` ← `silo.extract_service` (obligatorio; también es el `func` base).
- `keyword` ← `silo.keywords_service` (opcional; ojo: clave **singular**
  `"keyword"`, no `"keywords"` — `adapters.py`).
- `vlm` ← `silo.vlm_service` (opcional).
- `query` **se omite a propósito** (siempre `only_need_context=True`, §7).

Cada `func` de rol se crea con `build_llm_model_func` (`adapters.py`), que
reusa `create_llm_from_service` (`tools/aiServiceTools.py`) con
`temperature=0.0`. La firma expuesta a LightRAG es
`async def(prompt, system_prompt=None, history_messages=None, **kwargs) -> str`
(`adapters.py`). Las llamadas se etiquetan con
`metadata={"lc_source": "lightrag"}` para que el filtro
`_INTERNAL_LC_SOURCES` no filtre el JSON de keywords al stream de chat
(`adapters.py`).

#### 9.1.1 Recomendación de modelos por rol

Benchmark comparativo (cloud vs. open-source, `extract` y `keyword`, corpus
de 4 manuales completos + benchmark de calidad): ver
[`docs/testing/lightrag_extraction_benchmark_corpus.md`](../testing/lightrag_extraction_benchmark_corpus.md).

**Dónde lo ve el usuario**: al crear/editar un Silo o Repository, el
desplegable de `extract_service`/`keywords_service` muestra estas
recomendaciones directamente en el formulario — componente
`LightRAGModelHints` (`frontend/src/components/forms/LightRAGModelHints.tsx`,
usado desde `SiloForm.tsx` y `RepositoryFormPage.tsx`), fila "Open-source" de
cada rol. Este benchmark es la evidencia que respalda esas dos filas
concretas; el resto de filas (OpenAI, Anthropic, Mistral, Google) no se ha
verificado con este mismo método.

Recomendación resultante, en producción vía `LIGHTRAG_EXTRACT_MODEL` /
`LIGHTRAG_KEYWORD_MODEL` (o el `AIService` del silo correspondiente):

- **`extract`**: **Qwen3-30B-A3B-Instruct** — el más exhaustivo de los
  probados sobre las 61 chunks del corpus, sin diferencias de exactitud
  factual frente a alternativas cloud. Requiere `--json` (ver 9.1.2) sin
  excepción — en modo delimitado es directamente inutilizable. Tiene
  variación real entre pasadas idénticas (mismo prompt, `temperature=0`);
  no tratar sus totales como una cifra fija si se usa para dimensionar un
  SLA de indexación.
- **`keyword`**: **Qwen3-4B-Instruct** — único modelo sin ningún defecto en
  el benchmark de 7 preguntas (frente a la alucinación puntual de
  Qwen3-30B-A3B y la verbosidad sistemática de gpt-5.4-nano), con respuestas
  idénticas a las de Qwen3-30B-A3B en varias preguntas — el modelo mayor no
  aporta calidad adicional en esta tarea concreta. Con solo 4B parámetros
  cabe holgadamente en la misma GPU que el modelo de `extract`, a diferencia
  de alternativas de 24B+ que compiten por VRAM.

Ambos se sirven con vLLM (`--served-model-name` **debe** coincidir
exactamente con el valor de `LIGHTRAG_EXTRACT_MODEL`/`LIGHTRAG_KEYWORD_MODEL`
— usar el nombre servido, no la ruta de HuggingFace del repo del modelo;
confundir ambos da un 404 silencioso solo detectable en tiempo de ejecución).

#### 9.1.2 `entity_extraction_use_json`: forzado en código, no en `.env`

El repo **fuerza `entity_extraction_use_json=True`** como kwarg explícito al
construir `LightRAG(...)` en `LightRAGStore._build_rag`
(`lightrag_store.py`), **independientemente** de la variable de
entorno `ENTITY_EXTRACTION_USE_JSON` que lee el propio default de LightRAG
(`false` si no está puesta). Motivo: en modo texto delimitado (el default
nativo de LightRAG) se documentaron dos fallos de pérdida silenciosa de
datos — Qwen3-30B-A3B generando cientos de relaciones inventadas sin
terminar nunca, y gpt-5.4-nano colapsando el formato de registro en contenido
repetitivo, perdiendo entidades y relaciones reales sin ningún error visible
(detalle completo en el benchmark enlazado arriba). Al pasarlo como kwarg de
constructor en vez de depender de la variable de entorno, ningún despliegue
(dev, cliente, CI) puede regresar accidentalmente al modo roto por un `.env`
incompleto.

> ⚠️ **Corrección** (versión anterior de este documento): se afirmaba que el
> modo JSON de LightRAG «al validar la estructura por esquema, no puede tener
> ninguno de los dos fallos por construcción». **Es falso.** El flag
> `entity_extraction_use_json` solo cambia el *prompt* (pide un objeto JSON) y
> el *parser* (`_process_json_extraction_result`, `operate.py`, que usa
> `json_repair`). No hay validación por esquema en el momento de generar, y
> nada acota la longitud de la salida. Evidencia: un log de vLLM con este modo
> activo muestra `guided_decoding=None` y una sola página de bibliografía
> generando ~16.000 tokens durante casi 3 minutos. Lo que el modo JSON sí
> aporta es que un fallo de formato es **visible** (el parser no encuentra
> registros) en vez de silencioso como en el modo delimitado.

#### 9.1.3 JSON forzado en el servidor (`response_format`) y tope de salida

Dos mecanismos independientes cubren los dos problemas que el modo JSON por sí
solo no resuelve. Ambos se aplican **únicamente al rol `extract`**:

| Qué | Dónde | Default |
|---|---|---|
| Esquema JSON enviado como `response_format` | `adapters.py` (`_EXTRACTION_JSON_SCHEMA`) | activo (`LIGHTRAG_EXTRACT_GUIDED_JSON=true`) |
| Tope de tokens de salida | `config.LIGHTRAG_EXTRACT_MAX_TOKENS` | sin tope |

**`response_format`**: el esquema replica exactamente el contrato que lee el
parser de LightRAG (`entities[{name,type,description}]`,
`relationships[{source,target,keywords,description}]`), así que el servidor
restringe la decodificación a algo que LightRAG sabe leer. Solo se manda a
proveedores con el parámetro `json_schema` de estilo OpenAI
(`_GUIDED_JSON_PROVIDERS`: `Custom`, `OpenAI`, `Azure`, `OpenRouter`);
Anthropic/Mistral/Google usan otros mecanismos y se quedan con el JSON de
prompt en vez de romper con un parámetro desconocido.

**Detalle crítico**: el func del rol `extract` **no se usa solo para extraer**
— `_handle_entity_relation_summary` (`operate.py`) lo reutiliza para
resumir descripciones de entidades en **texto plano** al fusionar. Forzarle el
esquema corrompería todas las descripciones del grafo. LightRAG consume
`_priority` antes de llamar (`utils.py`), así que la llamada no trae
ningún marcador explícito y el único discriminante disponible es el prompt.

`_derive_json_extraction_marker` (`adapters.py`) **no hardcodea** esa frase: la
deriva de la librería instalada en tiempo de construcción — coge la línea más
larga de `PROMPTS["entity_extraction_json_system_prompt"]` que no contenga
llaves (las llaves son placeholders o literales JSON escapados que no
sobreviven al templating) y que **no** aparezca en
`PROMPTS["summarize_entity_descriptions"]`. Así, si upstream reescribe el
prompt, el marcador se mueve con él en vez de dejar de coincidir en silencio.

Dos garantías de seguridad:

1. **Falla hacia el lado inofensivo.** Si no se puede derivar ninguna frase
   única (prompt ausente, o todas las candidatas colisionan con el prompt de
   resumen), `_resolve_json_marker` devuelve `None` y no se manda esquema a
   nadie: se vuelve al JSON de prompt, nunca a resúmenes corrompidos.
2. **La decisión se loguea.** `"LightRAG guided JSON on (marker: ...)"` o
   `"... off: <motivo>"` al construir la instancia, así que una degradación es
   visible en el log del backend en vez de invisible.

Verificado en vivo contra Qwen3-30B-A3B en vLLM, con el mismo func de rol:
llamada de extracción → JSON que pasa `json.loads` estricto; llamada de
resumen → texto plano en el idioma pedido, sin rastro de JSON.

**Tope de salida**: sin `max_tokens`, un servidor OpenAI-compatible como vLLM
usa «lo que queda de ventana» (~30k tokens). Cortar no pierde el trabajo hecho:
el parser usa `json_repair`, que cierra las estructuras abiertas y conserva
todos los registros completos anteriores al corte (verificado).

### 9.2 Embeddings

`build_embedding_func` (`adapters.py`) construye un `EmbeddingFunc`
(`lightrag/utils.py`) desde `silo.embedding_service`, reusando
`get_embeddings_model` (`tools/embeddingTools.py`). Detecta la dimensión con una
llamada de sonda a `"."` (`adapters.py`). `max_token_size` sale de una
tabla por modelo (`_EMBEDDING_MAX_TOKENS_BY_MODEL`, `adapters.py`; default
`8192`, `adapters.py`). Campos de `EmbeddingFunc`: `embedding_dim`, `func`,
`max_token_size`, `model_name` (`utils.py`).

### 9.3 Métricas de tokens

Durante la indexación, cada llamada LLM/embedding reporta tokens a un
`IndexingTokenAccumulator` (`token_accumulator.py`) vía un `contextvar`
(`adapters.py`: `build_llm_model_func` para LLM, `build_embedding_func` para embeddings). Prioridad de
fuente: `usage_metadata` de LangChain → `response_metadata.token_usage` →
estimación con tiktoken (`adapters.py`). Los totales se guardan en la
tabla `indexing_metric` (`models/indexing_metric.py`), un row por run
(`prompt_tokens`, `completion_tokens`, `total_tokens`, `tokens_source`,
`llm_calls`, `embedding_tokens`, `duration_seconds`, `cost`, `model_name`).

---

## 10. Cómo lo usa específicamente este repositorio

### 10.1 Wiring general

- **Factory**: `VectorStoreFactory._create_lightrag_backend`
  (`tools/vector_store_factory.py`) es el único sitio que instancia
  `LightRAGStore`. Comprueba `is_lightrag_available()` y exige
  `ai_service` + `embedding_service`. `IMPLEMENTED_TYPES` incluye `'LIGHTRAG'`
  (`vector_store_factory.py`).
- **Un silo LightRAG = un workspace** `silo_{id}`. Se configura por-silo (§3.2)
  desde `SiloService` (`services/silo_service.py`) y `RepositoryService`
  (`services/repository_service.py`).

### 10.2 Indexación

`SiloService` → `LightRAGStore.index_documents` (§5). Los documentos vienen de
`Repository` (ficheros) o `Domain` (scraping) vectorizados en el silo. Métricas
en `indexing_metric`.

### 10.3 Recuperación en la ejecución del agente

- El agente obtiene un `LightRAGRetriever` vía `get_retriever` /
  `get_retriever_tool` (`tools/agentTools.py`). El modo se resuelve
  desde `agent.lightrag_query_mode` (`agentTools.py`); `skill-routed` es
  un caso especial (`agentTools.py`).
- **Citas inline**: `_append_lightrag_citation_sources`
  (`agentTools.py`) numera los chunks de `lightrag_raw_data.data.chunks`
  y pide al LLM citar como `[N](cite://N)`. La numeración es global en el turno
  (patrón de `offset` mutable) para soportar multi-silo.
- **Chunks de cobertura con `id`**: `list_documents_mentioning` construye sus
  chunks con `_coverage_chunk` (`agentTools.py`), que publica el `id` de LightRAG
  del chunk (columna `lightrag_doc_chunks.id`, leída por `find_chunks_mentioning`
  y propagada en el cuarto elemento de cada snippet). El playground marca en verde
  un chunk citado y las entidades cuyo `source_id` lo contiene comparando ese `id`;
  sin él, las citas de las listas de cobertura no se pintan. Un chunk de cobertura
  y su gemelo de `retrieve_from_knowledge_base` pueden compartir `id` en un mismo
  payload: la numeración sigue siendo posicional y los chips usan el índice en su
  clave de React.
- **Streaming del grafo**: `agent_streaming_service.py` acumula los
  eventos `_lightrag_graph` del turno con `merge_lightrag_graph`
  (`tools/streaming_utils.py`) y los emite como `lightrag_graph` en la respuesta
  final (`agent_streaming_service.py`) → el frontend los pinta como burbuja
  de grafo (`LightRAGGraphBubble`).

### 10.4 Endpoint de grafo del silo

`GET .../silos/{id}/graph` → `get_silo_graph` (`routers/internal/silos.py`)
→ `SiloGraphService.get_silo_graph` (`silo_graph_service.py`). **Siempre usa
Cypher directo** (no la API `get_knowledge_graph` de LightRAG) porque LightRAG
guarda el workspace como **label**, no como propiedad, y su API filtra por
propiedad → devolvería vacío (`silo_graph_service.py`). Filtra por label
`silo_{id}` con backticks, cuenta solo nodos con `entity_id` (excluye chunks),
y ordena por grado (`silo_graph_service.py`).

### 10.5 Manejo del event loop (concurrencia)

Detalle no obvio pero crítico: Neo4j (driver async) liga sus futures a un loop
concreto. El repo usa `_aget_rag_instance` en la ruta de query para inicializar
Neo4j **en el mismo loop** que corre `aquery_llm`, evitando "Future attached to a
different loop" (`lightrag_store.py`). `_run_async`
(`lightrag_store.py`) reusa el loop del hilo en contexto síncrono y
descarga a un hilo worker si ya hay un loop corriendo.

### 10.6 Ciclo de vida de las instancias `LightRAG` (conexiones)

`VectorStoreFactory` construye un `LightRAGStore` nuevo en cada llamada, pero las
instancias `LightRAG` que construye (driver de Neo4j, cliente de Qdrant, sonda de
embeddings; 1-13 s) **se comparten por proceso**: viven en `_SHARED_RUNTIME`
(`lightrag_store.py`), una por colección (`silo_{id}`) y worker, con su loop
dedicado. Los stores creados desde el factory (`shared_runtime=True`) las reutilizan
mientras coincida la huella de configuración (`_fingerprint`: servicios LLM/embeddings
y ajustes del silo). Si se edita el silo, la siguiente consulta reemplaza la instancia
y la vieja se finaliza (`finalize_storages`) tras `_RETIRE_GRACE_S` (900 s, para que
termine un turno en curso). `delete_collection` también la cierra.

Antes, cada llamada construía y abandonaba su instancia: en ~3 h de uso el backend
llegó a 1121 inicializaciones, 100 sockets abiertos y cientos de directorios
`/tmp/lightrag_*`. Un store con servicios que no son filas ORM (tests) conserva el
comportamiento privado anterior (se cierra al recolectarse).

Presupuesto de conexiones a Postgres: con `UVICORN_WORKERS=8` y los pools por defecto,
solo DBOS (4/worker) y el checkpointer (1/worker) ocupan 40 de las 100 de
`max_connections`; cada worker añade sus pools de SQLAlchemy según la concurrencia.
Con 4 consultas pesadas en paralelo se midió un pico de 85. Para consultas con mucha
concurrencia, bajar `UVICORN_WORKERS`, `DB_POOL_SIZE`/`DB_MAX_OVERFLOW` o subir
`max_connections` de Postgres.

---

## 11. Checklist de verificación rápida

| Afirmación | Comando / archivo |
|---|---|
| Versión que corre = `1.5.6` | `docker compose run --rm backend python -c "import importlib.metadata as m; print(m.version('lightrag-hku'))"` |
| **Al subir versión**: el JSON guiado sigue activo | Buscar en el log del backend `LightRAG guided JSON`: debe decir `on (marker: ...)`. Un `off:` indica que el marcador ya no se puede derivar (ver 9.1.3) |
| Versión fijada = `1.5.6` | `backend/Dockerfile:75` |
| Servicio = `backend` | `docker/docker-compose.yaml:42` |
| Feature flag | `config.LIGHTRAG_ENABLED` + `is_lightrag_available` (`adapters.py`) |
| Instanciación de LightRAG | `LightRAGStore._build_rag` (`lightrag_store.py`) |
| Pipeline de inserción | `LightRAGStore.index_documents` → `_ainsert` (`lightrag_store.py`) |
| Extracción de entidades | `extract_entities` (`lightrag/operate.py`, contenedor) |
| Queries (`only_need_context`) | `LightRAGRetriever` (`lightrag_store.py`) |
| Storage backends | `build_storage_config` (`storage_config.py`) |
| Bug `_cleanup_postgres` | `_cleanup_postgres` (`lightrag_store.py`) vs `NAMESPACE_TABLE_MAP` (`kg/postgres_impl.py`) |
| LLM por rol | `build_role_llm_configs` (`adapters.py`) |
| Embeddings | `build_embedding_func` (`adapters.py`) |
| Grafo por silo | `SiloGraphService.get_silo_graph` (`silo_graph_service.py`) |

---

## 12. Limitaciones conocidas de recuperación

Encontradas midiendo el corpus DOMUSA (benchmark ES/GB, 33 manuales en PDF). Son
límites de los datos o de la indexación, no fallos del agente; se anotan para no
volver a investigarlos.

### 12.1 Lo que no es texto no se indexa

La extracción de texto del PDF solo conserva texto. Se pierde, y la recuperación
no puede responder sobre ello:

- **Gráficas e imágenes**: de una gráfica solo queda su título. «¿Para qué modelos
  hay gráfica de prestaciones?» solo ve los títulos (p. ej. DSAT000120 pp.165-167).
- **Marcas gráficas de tablas** (✓ / ● en una matriz modelo × función): la celda
  queda vacía.
- **PDF escaneados**: sin capa de texto la extracción no devuelve nada que indexar
  (caso EN-UC4-1 del benchmark). No se ha probado el rol `vlm` como vía de OCR.

### 12.2 Una lista de figuras o una sección que se corta entre páginas — RESUELTO

**Historial** (válido para silos aún no reindexados con el esquema por-fichero
de §13, o indexados antes de él): cada página del PDF se indexaba como un
documento LightRAG propio (`res{id}-p{n}`) y, con `chunk_size` 2000, casi
nunca se dividía: en el silo 37, 1.600 páginas → 1.600 chunks. El
`chunk_overlap` solo actuaba **dentro** de una página, así que subirlo no
hacía nada contra esto: nunca cruzaba el borde de página. Efecto observado: si
el título de la última figura de una sección caía arriba de la página
siguiente, pegado a otra sección (DSAT000120 p.167: «Dual Clima 19R/19RT»
seguido de «12.4 Comprobación de cantidad de refrigerante»), la búsqueda
semántica no la asociaba a la sección anterior y el agente se dejaba ese
elemento.

**Estado actual**: desde el refactor de documento-por-fichero (§13), un PDF es
UN único documento LightRAG y el chunker ve el texto completo con marcadores
de página (§13.4); un chunk puede cruzar la frontera de página con
normalidad, igual que cualquier otro tramo del texto. La cita que recibe el
agente usa la página donde EMPIEZA el chunk (`p{P}` en
`res{N}-p{P}-chunk-NNN`), no un corte forzado por página. Sigue pendiente de
reindexar en los silos que todavía tengan el esquema antiguo — ver
"Migración" en §13.8.

### 12.3 La búsqueda literal exige las palabras de la página

Una ficha para rellenar que se llama «PARAMETERISATION» no se encuentra buscando
«parameter sheet»; una hoja «START-UP PROCEDURE» no comparte ninguna palabra con
«ficha de parametrización». Su huella estructural (huecos `………`, casillas `□`)
sí es exacta en el corpus, pero solo la usaría un humano. Sin cambio previsto.

### 12.4 Texto corrompido por ligaduras (DSAT000008)

La capa de texto del manual de servicio EVOL-TOP (DSAT000008, 103 páginas) pierde
las ligaduras «fi» y «fl», que salen como `ö` y `ø` dentro de las palabras
(«ediö cio», «Especiö caciones», «ø ujo»): 35 páginas con `ö` en medio de palabra,
y es el único documento del corpus con el defecto (en los demás la `ø` es el
símbolo de diámetro, siempre seguida de un número). Efecto: buscar
«especificaciones», «edificio» o «flujo» no recupera ese manual, ni por búsqueda
literal ni por la unión de cobertura. Las palabras sin «fi»/«fl» (caldera, agua,
gas…) sí lo encuentran, así que un sondeo con palabras corrientes no lo delata.

Arreglo posible, **no hecho**: reparar el texto al extraerlo del PDF (`ö` dentro de
una palabra → `fi`; `ø` entre letras → `fl`, sin tocar `ø` + dígito) y reindexar solo
ese documento. Comprobar antes sobre el texto reparado que no queden otras ligaduras
(«ff», «ffi»).

### 12.5 Preguntas de auditoría: revisar un elemento de todos los documentos

Una pregunta como «¿algún documento tiene la portada con errores de idioma?» (benchmark
ES UC4-5; la respuesta era una errata francesa, «INSTRUCCIONS D´INSTALLATION», en la
portada bilingüe de CDOC001372) no tiene ningún término que buscar. Exige leer un
elemento fijo (la página 1) de los 33 documentos y detectar una anomalía, y no hay
una herramienta que devuelva «la primera página de cada documento» (la más cercana,
`doc=`, son 33 llamadas). La recuperación por pasajes está pensada para localizar
contenido, no para auditar. Mismo tipo que §12.4: se contesta con una herramienta o
un script de revisión, no con el agente. Sin cambio previsto.

---

## 13. Esquema de almacenamiento (un documento por fichero)

Desde el refactor `lightrag-pdf-document-schema`, un fichero de Resource
genera **un único documento LightRAG por fichero**, no uno por página. Cambia
el shape de los ids de documento/chunk, cómo se reconstruye la etiqueta de
página y cómo se mide el progreso; el resto del pipeline (§5) no cambia.

### 13.1 Id de documento

`extract_documents_from_file` (`silo_service.py`, ruta `split=False`) devuelve
EXACTAMENTE un `Document` por fichero, cualquiera que sea su extensión. Para
un fichero ligado a un `Resource`, ese `Document` lleva `id=res{N}` grabado en
el momento de la extracción (`Document.id`, desde
`base_metadata['resource_id']`). `_lightrag_doc_id` (`lightrag_store.py`)
confía en ese id **solo** si tiene la forma exacta `res\d+` — ningún metadato
suministrado por un caller puede llegar a producir ese id:
`SiloService._create_documents_for_indexing` reconstruye siempre los
`Document`s desde dicts sin campo `id`, así que esta vía nunca es alcanzable
con datos no confiables (la comprobación de forma es defensa en profundidad,
no un hueco explotable hoy).

### 13.2 Formas de id de chunk

LightRAG deriva el id de cada chunk prefijando el id de documento:
`{doc_id}-chunk-{NNN}` (y `-sNN` si el chunker nativo tuvo que volver a partir
el chunk por tamaño — "hard split"). Según el origen del fichero:

| Origen | Id de documento | Id de chunk |
|---|---|---|
| PDF (Resource) | `res{N}` | `res{N}-p{P}-chunk-NNN[-sNN]` — `P` es la página donde EMPIEZA el chunk (un chunk puede cruzar páginas, ver §13.4) |
| DOCX/TXT/MD (Resource) | `res{N}` | `res{N}-chunk-NNN` (sin página) |
| No-Resource (p. ej. scraping de `Domain`) | hash de contenido (`compute_mdhash_id`, LightRAG) | `doc-<hash>-chunk-NNN` |
| Legacy, aún posible | `res{N}-p{P}` (un documento por página) | `res{N}-p{P}-chunk-NNN` |

El shape "legacy" **no ha desaparecido**: sigue produciéndose cuando el
metadato de un caller trae a la vez `resource_id` **y** `page` en una ingesta
`split=True` (algunas rutas de API pública/SharePoint/media del playground lo
hacen así) — comportamiento sin cambios, documentado aquí para que no se
confunda con un bug. `_lightrag_doc_id` (`lightrag_store.py`) resuelve ese
caso a `res{id}-p{page}` igual que siempre lo hizo.

### 13.3 Marcadores de página (`<<<page:N>>>`)

Un PDF extraído como `Document` único lleva el texto de cada página unido con
una línea marcador propia, generada por `page_marked_text` (`lightrag_store.py`):
una línea completa `<<<page:N>>>` seguida de exactamente un `\n`. Solo cuenta
como texto paginado el que EMPIEZA por un marcador (`strip_page_markers`,
`lightrag_store.py`) — un marcador-lookalike en mitad de otro texto nunca se
trata como frontera de página real.

Cada página se neutraliza con `neutralize_page_text` (`lightrag_store.py`)
antes de unirse, para que una línea del propio documento que SE PAREZCA a un
marcador no pueda confundirse luego con uno real — incluye el re-escapado de
`&` tras el sanitizado de LightRAG, que cierra una vía de forjado por
doble-decodificación de entidades HTML (detalle completo en el docstring de
`neutralize_page_text`). Los ficheros no-PDF (DOCX/TXT/MD) también pasan por
`neutralize_page_text` aunque no lleven marcadores de página, porque también
reciben un id `res{N}` y por tanto son alcanzables por los mismos lectores
basados en regex.

### 13.4 Chunking consciente de página (`chunking_func`)

`_achunk_page_aware` (`lightrag_store.py`), instalado como el `chunking_func`
legacy de LightRAG en `_build_rag`, corre la estrategia de chunking del silo
(`F`=fixed-token, `R`=recursive-character, `V`=semantic-vector,
`P`=paragraph-semantic, resueltas vía `_resolve_strategy_options`,
`lightrag_store.py`) **una sola vez sobre el fichero entero** (ya sin
marcadores, extraídos previamente con `strip_page_markers`), y después
atribuye cada chunk resultante a una página **bisecando** sobre los offsets de
marcador (`bisect_right`, `lightrag_store.py`). El selector explícito de
estrategia de chunking que ya tenía LightRAG (la ruta `process_options`/
`resolve_chunk_options` de §5) sigue existiendo para elegir QUÉ estrategia
usar, pero la función de chunking en sí —texto marcado o no— pasa siempre por
este hook.

Consecuencia directa: un chunk **puede cruzar una frontera de página**, igual
que cruzaría cualquier otro punto intermedio del texto — ya no hay un tope
duro de "nunca dos páginas en el mismo chunk" como en el esquema por-página
anterior (§12.2, resuelto). La página que se reporta en el chunk id es la de
INICIO del chunk, no todas las que toca.

LightRAG sigue logueando su propia línea genérica `"Chunking F(legacy)"`
(siempre, cualquiera que sea la estrategia real — es el log nativo del hook
legacy); el repo añade encima su propia línea, más precisa:
`"[workspace] Page-aware chunking <opción>: N chunk(s), M page marker(s)"`
(`_achunk_page_aware`, `lightrag_store.py`). Para saber qué estrategia corrió
de verdad, mirar esta segunda línea, no la de LightRAG.

### 13.5 Reconstrucción de la etiqueta "(p. N)"

La etiqueta legible que ve el usuario (`fichero.pdf (p. 12)`) ya no viaja en
el metadato del `Document` — un documento de un solo fichero no tiene un
`page` fijo que adjuntarle (`_source_label_from_metadata`,
`lightrag_store.py`). En su lugar se reconstruye EN LECTURA, a partir del id
del chunk, con `_page_label` (`lightrag_store.py`): toma el nombre base y, si
hay página, le añade `(p. N)` — de forma idempotente (una etiqueta antigua que
ya terminaba en `(p. N)` no se duplica). Los lectores que antes leían
`metadata["page"]` ahora extraen la página del propio chunk id vía
`CHUNK_ID_RESOURCE_PAGE_RE` (`lightrag_store.py`).

### 13.6 Progreso por chunks

La barra de progreso de indexación cuenta CHUNKS, no documentos/páginas (antes
un recurso de 90 páginas saltaba de 0 a 100% en un solo paso al terminar su
único documento LightRAG). `_doc_chunk_progress` (`lightrag_store.py`) lee,
por cada documento abierto, su `chunks_count` (total, fijado por LightRAG en
cuanto trocea el fichero) y cuenta cuántos de esos chunks ya tienen
`llm_cache_list` no vacío (extracción de entidades/relaciones ya ejecutada)
como numerador. Un documento con `chunks_count == 0` (aún no troceado) no
aparece en el resultado — los callers deben tratar la ausencia como "total
desconocido", nunca como cero. Es una query sync aparte (siempre vía
`asyncio.to_thread`) que propaga cualquier fallo; quien la llama
(`_aprocess_enqueued_with_progress` / `_report`) es quien la trata como
best-effort y degrada a la estimación por-fichero si falla. El progreso se
recorta para no mostrar el 100% hasta que el documento esté realmente en
estado terminal.

**Observabilidad**: un fallo de esta query (p. ej. un futuro bump de
`lightrag-hku` que renombre `chunks_list`/`llm_cache_list`/`chunks_count`)
loguea `"[workspace] Chunk-progress query failed (best-effort): ..."` como
WARNING una vez por corrida (el resto de repeticiones cae a DEBUG para no
inundar el log cada 0.5s) — `_aprocess_enqueued_with_progress`
(`lightrag_store.py`). Es el canario de que la barra ha degradado, en
silencio y para siempre, a la estimación por-fichero; una alerta sobre esa
línea es la señal más temprana de que esta query dejó de funcionar.

### 13.7 Compatibilidad hacia atrás

Todos los lectores que tocan ids de documento/chunk aceptan las DOS formas
(antigua por-página y nueva por-fichero) a la vez: `_lightrag_doc_id`,
`CHUNK_ID_RESOURCE_PAGE_RE`, `_FULL_DOC_ID_RESOURCE_RE`, `_page_label`,
`_group_chunk_rows` y los filtros SQL de `find_chunks_mentioning` (todos en
`lightrag_store.py`) reconocen `res{N}-p{P}(-chunk-...)` y
`res{N}(-chunk-...)` indistintamente. Un silo con documentos antiguos y
nuevos mezclados (ver migración abajo) funciona sin degradar ninguna
funcionalidad de lectura.

### 13.8 Migración

**La migración es OPCIONAL.** Un silo con documentos del esquema antiguo
sigue funcionando tal cual, sin ninguna acción: §13.7 ya garantiza que todos
los lectores entienden ambos shapes a la vez. Migrar solo aporta un beneficio
concreto — que los PDFs reindexados dejen de tener el límite de "nunca cruza
una página" en el chunking (§12.2, resuelto) — nada se rompe si no se migra.

#### Comprobar si hay una indexación en vuelo

Antes de tocar nada: `GET /internal/apps/{app}/repositories/{id}/ingestion-status`
→ `is_indexing` (`backend/routers/internal/repositories.py`), respaldado por
el advisory lock del silo — es `true` solo mientras una corrida está REALMENTE
viva, no solo porque queden filas sin terminar. Complementario a nivel de
fila: `resource.status IN ('pending', 'indexing', 'paused')` sobre los
recursos del repositorio.

> `is_indexing` solo cubre a quien TOMA `silo_indexing_lock`: la ingestión de
> Resources, el reindexado de un único fichero y la ingesta de Azure Blob
> (`routers/internal/silos.py`, `resource_service.py`,
> `azure_blob_ingest_service.py`). **No** cubre los tasks de Media, el
> scraping de `Domain` ni la ingesta pública `index-file`, que escriben al
> mismo workspace del silo sin tomar ese lock — `is_indexing: false` no
> garantiza que nada esté escribiendo en el silo ahora mismo si cualquiera de
> esas otras vías puede estar activa.

#### Camino NO destructivo (solo para silos 100% Resource): vaciar el silo sin perder Repository/ficheros

```
DELETE /public/v1/app/{app}/silos/{id}/docs/delete/all
```

(`backend/routers/public/v1/silos.py` → `SiloService.delete_all_docs_in_collection`,
que llama a `delete_collection` del store — §5, "Borrado"). Limpia Neo4j,
Qdrant y PostgreSQL del workspace `silo_{id}` **entero** — TODO lo que viva en
ese workspace, no solo los documentos de `Resource`: los chunks de `Media`
(`SiloService.index_media_chunk` escribe en la misma colección del silo), el
contenido metido por la ruta pública `index-file` y los chunks de `Domain`
también desaparecen. Es "no destructivo" únicamente respecto al `Silo`, el
`Repository` y los ficheros subidos (que sí sobreviven, a diferencia de
borrar el Repository o el Silo enteros — ver "Qué SÍ se pierde" más abajo):
**solo es seguro lanzarlo contra un silo cuyo contenido sea enteramente
ficheros de `Resource`** — contra un silo con Media/Domain/`index-file`
mezclado, borra ese contenido exactamente igual que borrar el Silo entero.
**Solo existe como endpoint de la API pública — no hay botón equivalente en
la UI interna.**

> ⚠️ Esta ruta **no toma el `silo_indexing_lock`** del silo
> (`services/silo_service.py`, el mismo lock que usa la indexación) — nada le
> impide correr en paralelo con una ingestión activa, lo que dejaría el
> workspace a medio vaciar sobre trabajo en curso. Confirmar `is_indexing:
> false` primero y no lanzarlo si alguien puede arrancar una ingestión a la
> vez (p. ej. bloquear la ventana de mantenimiento).

Después de vaciar, hay que **reindexar cada `Resource`** para que vuelva a
haber contenido:

- **Repos pequeños (unas pocas decenas de ficheros)**: uno a uno,
  `POST /internal/apps/{app}/silos/{id}/resources/{resource_id}/reindex`
  (`backend/routers/internal/silos.py`) — síncrono, un fichero cada vez.
- **Repos grandes**: poner `resource.status = 'pending'` en las filas
  afectadas y pulsar "Resume indexing" en la UI (usa el alimentador con
  ventana deslizante, la barra de progreso y pausa de siempre, §5.1/§13.6).
  **`resource_service.py` confirma que `resume_indexing` solo recoge filas en
  `pending/error/indexing` + `STOP_STATUSES` (`paused`, `cancelled`) — NO
  `ready`** (`backend/services/resource_service.py`), así que un simple
  "Resume" sobre recursos ya `ready` no hace nada: hace falta resetear su
  `status` a `pending` primero, a mano (no hay endpoint para esto). El
  modelo usa `__tablename__ = 'Resource'` (con mayúscula — Postgres es
  sensible a mayúsculas en identificadores entrecomillados, así que el
  nombre va entre comillas dobles):
  ```sql
  UPDATE "Resource" SET status = 'pending'
  WHERE repository_id = :id AND status = 'ready';
  ```
  **Verificar este paso en staging antes de darlo por un procedimiento
  cerrado** — no se ha probado en vivo contra una tabla real de producción.

#### Qué SÍ se pierde si en vez de esto se borra el Repository o el Silo

Borrar el `Repository` elimina también los ficheros subidos
(`repository_service.py`, `RepositoryService.delete_repository`: borra cada
fichero físico y la carpeta del repositorio antes de borrar las filas).
Borrar el `Silo` elimina además sus `Media` (migraciones de vídeo/audio, `ON
DELETE CASCADE`) y sus ficheros (`SiloService.delete_silo` →
`remove_silo_media_files`, `services/silo_service.py`). Contenido que **no**
se puede volver a alimentar después de un borrado así:

- **Media** ya indexada: el fichero se borra en cuanto termina de indexarse
  (comportamiento documentado del pipeline de vídeo/audio) — no queda nada
  que resubir.
- Contenido metido por la ruta pública `index-file` sin pasar por un
  `Resource`: no se conserva como `Resource`, así que no hay de dónde
  reextraerlo.
- Rastreos de `Domain` (scraping): exigirían un recrawl completo.

**Ninguno de estos tres tipos de contenido usa el shape `res{N}` nuevo** (su
id sigue siendo `doc-<hash>`, §13.2) — vaciar/borrar para "migrarlos" no
aporta nada y los pierde para siempre, ya sea con `delete/all` o borrando el
Silo/Repository entero: ambos caminos destruyen exactamente el mismo
contenido no-Resource (ver el aviso de la sección anterior). **Un silo con
Domain/Media/`index-file` mezclado con ficheros de Resource no tiene un
camino de migración seguro con las herramientas de hoy** — ni vaciarlo ni
borrarlo. Si de verdad se quiere el chunking nuevo para ESE silo, la única
opción sin pérdida es la ruta de Silo/Repository paralelo del apartado de
coste/tiempo de inactividad, más abajo: indexar los ficheros de Resource ahí
con el esquema nuevo y dejar el Domain/Media/`index-file` original donde
está. Si no, recordar que migrar es opcional (arriba): lo más seguro es
simplemente no tocar ese silo.

#### Pausar, cancelar o desplegar en mitad de una ingestión

**Preferir dejar terminar las ingestiones en curso** antes de desplegar este
cambio, en vez de pausar o cancelar. Un despliegue (reinicio del proceso)
interrumpe cualquier ingestión activa por sí solo — esto es un hecho a tener
en cuenta, no un procedimiento nuevo: el hilo en background que la llevaba
muere con el proceso anterior.

1. **"Cancel" borra datos.** `_discard_unindexed_after_cancel`
   (`backend/services/resource_service.py`) borra fila **y fichero** de
   cualquier recurso cancelado que no llegara a indexar ni un documento. Solo
   usar `cancel` si se acepta perder (y tener que resubir) esos ficheros
   concretos; no es una operación inocua.
2. **Cualquier recurso no-`ready` con páginas antiguas ya indexadas puede
   duplicarse al reanudarlo tras el despliegue** — no solo el caso de
   "pausar antes del despliegue". Las mismas filas quedan expuestas por una
   vía más común: al arrancar, `reset_stuck_resources`
   (`backend/startup_tasks.py`) pasa a `error` cualquier recurso que
   estuviera `indexing`/`pending` cuando el proceso anterior murió — y el
   propio botón "Resume" de la UI invita a reanudar justo esos recursos
   `error` (`resume_indexing` los recoge, ver arriba). Un PDF interrumpido A
   MITAD, por cualquiera de las dos vías (pausa manual o reinicio del
   despliegue), deja páginas ya indexadas como documentos `res{N}-p{P}`
   (shape viejo); reanudarlo ya con el código nuevo desplegado hace que el
   resto del fichero entre por la vía nueva y genere, además, un documento
   `res{N}` separado para el fichero entero — el mismo caso "reindexar un
   único fichero antiguo lo duplica" (§13.2), disparado automáticamente por
   el propio resume.

   **Tras desplegar, no reanudar (ni reindexar) ningún recurso que no esté
   `ready` y que ya tenga páginas antiguas procesadas** — comprobarlo antes
   de pulsar "Resume":
   ```sql
   SELECT * FROM lightrag_doc_status
   WHERE workspace = 'silo_<id>' AND id LIKE 'res<N>-p%' AND status = 'processed';
   ```
   Si devuelve filas para ese recurso, mejor cancelarlo (asumiendo la
   pérdida del punto 1) o dejarlo tal cual hasta decidir cómo migrarlo a
   mano, en vez de reanudarlo sin más.

La frase "mezclar ambos shapes en el mismo silo no es dañino" (§13.7) se
refiere a recursos DISTINTOS conviviendo con shapes distintos, no a este caso
de un mismo recurso cruzando la frontera pausa/interrupción→despliegue→resume,
que sí produce una duplicación real.

#### Coste y tiempo de inactividad de un reindexado completo

Antes de vaciar/reindexar un silo entero, `POST
/internal/apps/{app}/silos/{id}/estimate-indexing` da una estimación de
coste/tamaño (§5, "Pipeline de inserción"). Los
agentes que usan ese silo devuelven recuperación vacía o parcial mientras
dura el reindexado — para un corpus grande puede ser horas. Para migrar sin
tiempo de inactividad: montar un Repository/Silo paralelo, indexarlo con el
esquema nuevo, y apuntar los agentes a él cuando esté listo (a cambio de
volver a subir los ficheros).

### 13.9 Rollback

Volver a una versión del backend anterior a este refactor sigue funcionando
para los documentos nuevos (`res{N}`) **con recuperación**, pero con
**citas/procedencia degradadas**: el código antiguo lee página con
`CHUNK_ID_RESOURCE_PAGE_RE` asumiendo el shape `res{N}-p{P}-chunk-NNN`, y no
reconoce `res{N}-chunk-NNN` (DOCX/TXT/MD) ni interpreta correctamente la
página de un chunk de PDF que ya viene con el shape nuevo — el número de
página y la etiqueta de procedencia se pierden para esos documentos, aunque
el contenido se siga recuperando.

- **No hacer rollback mientras haya una ingestión en vuelo** — comprobar
  `is_indexing` primero (§13.8).
- **No usar "cancel" en la versión antigua del código contra trabajo
  pendiente que ya se generó como `res{N}` nuevo**: el guard de descarte de
  la versión antigua solo reconoce `LIKE 'res{N}-p%'` (el shape viejo), así
  que un documento `res{N}` ya indexado no coincide con ese patrón y el
  recuento de "documentos indexados" de ese recurso le sale 0 — el guard
  antiguo lo trata como si nunca se hubiera indexado nada. Consecuencia
  concreta, no solo un huérfano en el grafo: **borra la fila del `Resource` y
  su fichero subido** (su criterio para "nunca indexado" es "bórralo"),
  mientras el contenido que SÍ llegó a indexarse (el documento `res{N}`
  completo, con sus entidades/relaciones) se queda para siempre en
  Neo4j/Qdrant/Postgres sin ninguna fila que lo referencie ni ningún
  mecanismo que lo limpie.

### 13.10 Limitaciones conocidas

- **Dedup dentro del mismo lote** (pre-existente, sin relación con este
  refactor, **no corregido aquí**): el dedup-por-nombre interno de LightRAG
  probablemente descarta todo chunk salvo el primero que comparta etiqueta en
  las rutas de ingesta LightRAG con `split=True` (index-file público,
  SharePoint, media del playground) — un problema aparte, fuera de alcance.
- Un payload `split=True` con página en el metadato (la vía legacy de §13.2)
  sigue produciendo etiquetas `nombre (p. N)`, no nombres pelados — es el
  comportamiento existente y sin cambios de esa vía concreta, no una
  regresión de este refactor.
- **Reindexar un único fichero antiguo sin recrear/vaciar el silo (§13.8)
  probablemente solo duplica de verdad para PDFs.** Antes de este refactor,
  un DOCX/TXT/MD ya tenía id `doc-<hash>` y una etiqueta de fichero pelada —
  la MISMA etiqueta que tendría el nuevo documento `res{N}` de ese mismo
  fichero. El dedup-por-nombre de LightRAG (§5.2, fila "File name already
  exists") plausiblemente rechaza el reindexado como una fila `dup-*`
  fallida en vez de crear un duplicado real — **no verificado en vivo**. Si
  se confirma en staging, tratarlo como: PDF → duplica el fichero;
  DOCX/TXT/MD → probablemente no-op silencioso (rechazado por dedup). Hasta
  verificarlo, no dar la duplicación por garantizada en formatos no-PDF.

---

*Documento generado analizando el código instalado en el contenedor
`mattin-backend` y la capa de integración del repositorio. Ninguna afirmación de
internals se basa en memoria; todas se verificaron sobre la fuente instalada.
Las referencias son **fichero + símbolo, sin número de línea** (ver la
convención en §1): los símbolos sobreviven a los bumps, los números no.*
