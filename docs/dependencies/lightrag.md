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
| **Instalada en el contenedor en ejecución** | **`1.5.5rc1`** | `docker compose run --rm backend python -c "import importlib.metadata as m; print(m.version('lightrag-hku'))"` → `1.5.5rc1` |
| **Fijada por el Dockerfile** | `1.5.5rc1` | `backend/Dockerfile:75` — `pip install ... "lightrag-hku[offline-storage]==1.5.5rc1"` |
| **Referida por los comentarios del código** | `1.5.5rc1` | `backend/tools/vector_stores/lightrag_store.py:4`, `backend/tools/vector_stores/lightrag/adapters.py:3` |
| **Comentario obsoleto en pyproject** | `1.5.0rc3` (texto stale) | `pyproject.toml:174` |

> ✅ **Versión alineada**: la imagen desplegada corre `1.5.5rc1`, igual que el
> pin del `Dockerfile` (commit `0582699f bump(lightrag): upgrade lightrag
> version`). Todos los *internals* de este documento están verificados sobre
> `1.5.5rc1` (lo que se ejecuta hoy). El único texto de versión pendiente de
> limpiar es el comentario de `pyproject.toml:174`, que aún dice `1.5.0rc3`.

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
| `LIGHTRAG_ENABLED` | `false` | `backend/config.py:63` |
| `NEO4J_URI` | `None` | `backend/config.py:64` |
| `NEO4J_USERNAME` | `neo4j` | `backend/config.py:65` |
| `NEO4J_PASSWORD` | `None` | `backend/config.py:66` |
| `ENTITY_EXTRACT_MAX_GLEANING` | `0` | `backend/config.py:72-74` |
| `LIGHTRAG_EXTRACT_MAX_TOKENS` | sin tope | tope de salida del rol `extract` (ver 9.1.3) |
| `LIGHTRAG_EXTRACT_GUIDED_JSON` | `true` | manda el esquema como `response_format` (ver 9.1.3) |
| `MAX_PARALLEL_INSERT` | `3` | documentos (= páginas) en paralelo; leída por LightRAG al importar |
| `MAX_ASYNC_LLM` | `4` | llamadas LLM concurrentes por rol; leída por LightRAG al importar |
| `EMBEDDING_FUNC_MAX_ASYNC` | `8` | llamadas de embedding concurrentes |
| `SQLALCHEMY_DATABASE_URI` | — | usado como `config.DATABASE_URL` para PG KV/doc-status (`storage_config.py:96-99`) |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` | `storage_config.py:109-110` |
| `VECTOR_DB_TYPE` | `PGVECTOR` | usado por el silo, no por LightRAG directamente |

> Nota: el default del repo para el *gleaning* es `0` (`config.py:72`), mientras
> que el default nativo de LightRAG `1.5.5rc1` es `DEFAULT_MAX_GLEANING = 1`
> (`lightrag/constants.py:17`, campo `entity_extract_max_gleaning` en
> `lightrag/lightrag.py:349`).

### 3.2 Configuración por-silo (columnas en la tabla `Silo`)

Cada silo LightRAG se configura de forma independiente. Comprobar en
`backend/models/silo.py:31-71`:

| Columna | Tipo | Rol |
|---|---|---|
| `vector_db_type` | String | `'LIGHTRAG'` activa este backend (`silo.py:45`) |
| `lightrag_vector_db_type` | String, def. `'QDRANT'` | qué vector store usa LightRAG por dentro: `QDRANT` o `PGVECTOR` (`silo.py:49`) |
| `lightrag_chunk_strategy` | String | `fixed_token`/`recursive_character`/`semantic_vector`/`paragraph_semantic` (`silo.py:51`) |
| `lightrag_chunk_token_size` | Integer | tamaño de chunk (`silo.py:52`) |
| `lightrag_chunk_overlap_token_size` | Integer | solape (`silo.py:53`) |
| `lightrag_language` | String | idioma de extracción de entidades/keywords → `addon_params['language']` (`silo.py:59`) |
| `lightrag_entity_extract_max_gleaning` | Integer | iteraciones de *gleaning* (`silo.py:66`) |
| `lightrag_max_source_ids_per_entity` | Integer | tope de source-ids por entidad (`silo.py:67`) |
| `lightrag_max_source_ids_per_relation` | Integer | tope de source-ids por relación (`silo.py:68`) |
| `lightrag_entity_types` | Text | categorías de entidad, separadas por comas → se traduce a `addon_params['entity_types_guidance']` (`silo.py:71`, ver §6) |
| `lightrag_graph_context_enabled` | Boolean, def. `false` | expone el grafo como burbuja en el playground (`silo.py:70`) |
| `use_agent_as_query` | Boolean, def. `false` | usa el LLM del agente para sintetizar (`silo.py:71`) |

**LLMs por rol** (una `AIService` independiente por rol — `silo.py:31-43`):
`extract_service_id`, `keywords_service_id`, `query_service_id`,
`vlm_service_id`. `indexing_service_id` es un alias legacy que mapea a
`extract`. La validación de que un silo LightRAG requiere al menos un servicio
`extract` + `embedding_service` está en
`backend/services/silo_service.py:712-730`.

**Modo de query por-agente** (única perilla LightRAG específica del agente):
`Agent.lightrag_query_mode` (`backend/models/agent.py:80-83`), valores
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
| `LightRAGStore` | `lightrag_store.py:472` | Implementa `VectorStoreInterface`; un silo = un *workspace* `silo_{id}` |
| `LightRAGStore._build_rag` | `lightrag_store.py:544` | Construye la instancia `LightRAG(...)` (sin inicializar) |
| `LightRAGStore._get_rag_instance` | `lightrag_store.py:528` | Cache síncrona (indexación / hilos de fondo) |
| `LightRAGStore._aget_rag_instance` | `lightrag_store.py:605` | Cache async con lock por-colección; inicializa Neo4j en el loop del caller |
| `LightRAGRetriever` | `lightrag_store.py:203` | `BaseRetriever` de LangChain sobre `aquery_llm` |
| `build_llm_model_func` | `lightrag/adapters.py:160` | Envuelve una `AIService` en la firma que espera LightRAG |
| `build_role_llm_configs` | `lightrag/adapters.py:106` | `dict[str, RoleLLMConfig]` para roles `extract/keyword/vlm` |
| `build_embedding_func` | `lightrag/adapters.py:302` | Construye `EmbeddingFunc` desde una `EmbeddingService` |
| `is_lightrag_available` | `lightrag/adapters.py:80` | Feature flag: `LIGHTRAG_ENABLED` + `import lightrag` OK |
| `build_storage_config` | `lightrag/storage_config.py:72` | Elige backends + exporta env vars (`NEO4J_*`, `QDRANT_*`, `POSTGRES_*`) |
| `IndexingTokenAccumulator` | `lightrag/token_accumulator.py:29` | Suma tokens LLM/embedding de un run de indexación |
| `SiloGraphService` | `services/silo_graph_service.py:22` | Lee el grafo de un silo desde Neo4j (Cypher directo) |

### 4.2 En LightRAG instalado (`1.5.5rc1`)

Raíz: `/usr/local/lib/python3.11/site-packages/lightrag/`.

| Símbolo | Archivo:línea (en el contenedor, `1.5.5rc1`) | Rol |
|---|---|---|
| `LightRAG` (dataclass) | `lightrag/lightrag.py:265` | Clase principal; mezcla `_RoleLLMMixin`, `_StorageMigrationMixin`, `_PipelineMixin` |
| `LightRAG.initialize_storages` | `lightrag/lightrag.py:1276` | Inicializa los objetos de storage del workspace |
| `LightRAG.ainsert` | `lightrag/lightrag.py:1474` | API alto nivel de inserción |
| `apipeline_enqueue_documents` | `lightrag/pipeline.py:238` | Encola documentos en `doc_status` |
| `apipeline_process_enqueue_documents` | `lightrag/pipeline.py:943` | Procesa la cola (chunk + extracción) |
| `LightRAG.aquery` | `lightrag/lightrag.py:2123` | Wrapper compat → solo `str`/stream |
| `LightRAG.aquery_llm` | `lightrag/lightrag.py:2387` | Devuelve dict con contexto + grafo (lo que usa el repo) |
| `LightRAG.aquery_data` | `lightrag/lightrag.py:2181` | Solo recuperación (`only_need_context=True`) |
| `QueryParam` (dataclass) | `lightrag/base.py:83` | Parámetros de query |
| `extract_entities` | `lightrag/operate.py:3328` | Extracción de entidades/relaciones |
| `kg_query` / `naive_query` | `lightrag/operate.py` | Motores de query grafo / vectorial |
| `RoleLLMConfig` | `lightrag/llm_roles.py:63` | Config LLM por rol |
| `resolve_chunk_options` | `lightrag/parser/routing.py:470` | Resuelve opciones de chunking |
| `EmbeddingFunc` (dataclass) | `lightrag/utils.py:498` | Envoltorio de embeddings |

**Construcción de la instancia** (`lightrag_store.py:584-596`): el repo pasa
`working_dir` (tempdir), `workspace=silo_{id}`, `llm_model_func` (base/fallback),
`role_llm_configs`, `embedding_func`, los cuatro backends de storage,
`entity_extract_max_gleaning` y `**extra_kwargs` (chunk sizes + source-id caps
solo si están configurados). El `language` se inyecta mutando
`rag.addon_params['language']` **después** de construir, para no pisar los
defaults de LightRAG (`lightrag_store.py:597-602`).

---

## 5. Pipeline de inserción

**Punto de entrada del repo**: `LightRAGStore.index_documents`
(`lightrag_store.py:636`). Flujo:

1. Filtra docs vacíos; alinea `texts[]` con `file_paths[]` (etiqueta de origen
   por chunk vía `_source_label_from_metadata`, `lightrag_store.py:78`).
2. Activa un `IndexingTokenAccumulator` en un `contextvar`
   (`set_active_accumulator`, `adapters.py:257`) para contar tokens.
3. Llama a `_ainsert_with_progress` → `_ainsert` (`lightrag_store.py:104-163`).
4. Devuelve un dict de métricas (`totals()` de `token_accumulator.py:58`) que
   se persiste en la tabla `indexing_metric` (`models/indexing_metric.py`).

**`_ainsert`** (`lightrag_store.py:104`) **no** llama al `rag.ainsert` normal;
en su lugar usa la ruta de chunking "moderna" explícitamente:

```python
chunk_opts = resolve_chunk_options(rag.addon_params, process_options=process_options)
await rag.apipeline_enqueue_documents(texts, file_paths=..., process_options="F", chunk_options=chunk_opts)
await rag.apipeline_process_enqueue_documents()
```

Motivo (docstring en `lightrag_store.py:104-125`): pasar `process_options` evita
que LightRAG caiga en la ruta legacy de 6 args (`Chunking F(legacy)`) y usa el
contrato nuevo del chunker (`Chunking F`).

**Selector de estrategia de chunking** (`lightrag_store.py:70-75`,
`_CHUNK_STRATEGY_OPTION`): mapea la estrategia del silo al char de
`process_options` de LightRAG:

| Estrategia del silo | `process_options` | Constante LightRAG |
|---|---|---|
| `fixed_token` | `"F"` | `PROCESS_OPTION_CHUNK_FIXED` |
| `recursive_character` | `"R"` | `PROCESS_OPTION_CHUNK_RECURSIVE` |
| `semantic_vector` | `"V"` | `PROCESS_OPTION_CHUNK_VECTOR` |
| `paragraph_semantic` | `"P"` | `PROCESS_OPTION_CHUNK_PARAGRAH` |

Desconocido → `"F"`. Comprobar constantes en `lightrag/constants.py:305-333`.

**Dentro de LightRAG** (`1.5.5rc1`):
- `apipeline_enqueue_documents` (`pipeline.py:238`): genera IDs MD5 + dedup por
  contenido, crea el estado inicial en `doc_status`, filtra ya-procesados.
  `chunk_options` (cuando es `None`) se deriva de `addon_params['chunker']` vía
  `resolve_chunk_options` y se persiste en `full_docs[doc_id]['chunk_options']`.
- `apipeline_process_enqueue_documents` (`pipeline.py:943`): guard de un-solo-
  worker vía `pipeline_status["busy"]` bajo `pipeline_status_lock` (por-workspace,
  `kg/shared_storage.py`); trocea, ejecuta extracción de entidades por chunk,
  actualiza estado. Los documentos se procesan de 3 en 3
  (`max_parallel_insert`, env `MAX_PARALLEL_INSERT`), y dentro de cada uno los
  chunks de 4 en 4 (`llm_model_max_async`, env `MAX_ASYNC_LLM`).
- **Progreso**: `_ainsert_with_progress` (`lightrag_store.py:179`) poll-ea cada
  0.5s `rag.doc_status.get_status_counts()` y cuenta `processed + failed` menos
  un baseline tomado antes de empezar (los documentos de recursos anteriores del
  mismo silo ya están `processed`). **No** usa `pipeline_status["cur_batch"]`:
  ese contador se incrementa cuando un documento *entra* al semáforo
  (`pipeline.py:2155-2178`), no cuando termina, así que las primeras
  `max_parallel_insert` páginas parecían instantáneas y el ETA salía sesgado a
  la baja. `pipeline_status` no tiene ningún contador de completados.
- `resolve_chunk_options` (`parser/routing.py:470`) lee `addon_params['chunker']`,
  recorta al sub-dict de la estrategia seleccionada por `process_options` y
  devuelve una copia profunda independiente.

**Borrado**: `delete_documents` y `delete_documents_excluding` son **no-ops
deliberados** (`lightrag_store.py:696-730`, marcados `ponytail:`) — LightRAG no
soporta borrado per-documento del grafo. Reindexar es idempotente (doc_id =
hash de contenido); los chunks huérfanos son el techo conocido. El borrado real
es a nivel de silo: `delete_collection` (`lightrag_store.py:732`) limpia Neo4j,
Qdrant y PostgreSQL directamente (helpers `_cleanup_neo4j/_cleanup_qdrant/
_cleanup_postgres`, `lightrag_store.py:752-826`).

---

## 6. Extracción de entidades

**Función LightRAG**: `extract_entities` (`operate.py:3328`).

- Usa el LLM del rol **`extract`** (`global_config["role_llm_funcs"]["extract"]`,
  `operate.py:3344`) — en el repo, el que construye `build_role_llm_configs`
  desde `silo.extract_service` (`adapters.py:106`, `lightrag_store.py:558-563`).
- **Gleaning**: `entity_extract_max_gleaning` (`operate.py:3345`). En
  `1.5.5rc1` corre **como mucho una vez** (`run_gleaning = ... > 0`,
  `operate.py:3527-3528`); reejecuta el par user/assistant inicial con un prompt
  "continue". Se salta si el payload combinado excede `MAX_EXTRACT_INPUT_TOKENS`.
  El valor viene del silo (`lightrag_entity_extract_max_gleaning`) o del default
  global `config.ENTITY_EXTRACT_MAX_GLEANING` (`lightrag_store.py:578-582`).
- **Idioma**: `addon_params['language']` (`operate.py:3366`), inyectado por
  el repo desde `silo.lightrag_language` (`lightrag_store.py:601-602`).
- **Tipos de entidad**: se resuelven en un *prompt profile* vía
  `resolve_entity_extraction_prompt_profile(addon_params, ...)` (en `prompt.py`);
  el `entity_types_guidance` sale de `addon_params['entity_types_guidance']` si
  es un string no vacío, si no del default `PROMPTS["default_entity_types_guidance"]`.
  **LightRAG no acepta una lista plana** (`addon_params['entity_types']` no lo lee
  nada del pipeline) — solo el string de `entity_types_guidance`.
  El repo sobreescribe esto desde `silo.lightrag_entity_types` (texto separado por
  comas en "Advanced settings") vía `_build_entity_types_guidance()`
  (`lightrag_store.py:90-119`), que genera el mismo formato (frase + bullets, sin
  descripciones por tipo) que usa el default de LightRAG. Si el campo está vacío:
  con `lightrag_language == 'Spanish'` se inyecta una traducción propia de los 11
  tipos por defecto de LightRAG (`_SPANISH_DEFAULT_ENTITY_TYPES`, ver §3, LightRAG
  no localiza esto por sí mismo); en cualquier otro caso no se toca `addon_params`
  y se usa el default nativo en inglés.
- **Prompts** (`lightrag/prompt.py`, dict `PROMPTS`): delimitadores
  `PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"` (`prompt.py:12`) y
  `PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"` (`:13`);
  `entity_extraction_system_prompt` (`:54`), `entity_extraction_user_prompt`
  (`:125`), `keywords_extraction` (`:484`).
  Hay variantes JSON (`entity_extraction_json_*`) según
  `global_config["entity_extraction_use_json"]`.

**Grafo resultante**: cada nodo/relación se guarda en Neo4j con el nombre del
workspace (`silo_{id}`) como **label** del nodo (no como propiedad) — clave para
entender `SiloGraphService` (§9). Comprobar en `silo_graph_service.py:74-85` y
el helper de limpieza `_cleanup_neo4j` (`lightrag_store.py:766-771`,
``MATCH (n:`silo_X`) DETACH DELETE n``).

---

## 7. Queries

**`QueryParam`** (`base.py:83`). Campos relevantes:
- `mode`: `local|global|hybrid|naive|mix|bypass` — **default nativo `mix`**
  (`base.py:86`), pero el repo usa `hybrid` por defecto.
- `only_need_context: bool` (`base.py:95`) — **el repo siempre lo pone `True`**.
- `top_k` (`base.py:107`), `chunk_top_k` (`base.py:110`).

**Modos válidos y mapeo del repo** (`lightrag_store.py:65`, `_resolve_query_mode`
en `:456`):
- Modos nativos (`local/global/hybrid/naive/mix/bypass`) pasan tal cual.
- `similarity` / `similarity_score_threshold` → `hybrid`; `mmr` → `mix`;
  desconocido → `hybrid`.

**Ruta de query del repo**:
1. `LightRAGRetriever` (`lightrag_store.py:203`) — retriever de LangChain.
   Siempre llama con `only_need_context=True` (`lightrag_store.py:241-245`), de
   modo que **LightRAG NO genera la respuesta**: devuelve solo el contexto del
   grafo y el LLM del agente sintetiza. Por eso el rol `query` se deja sin
   configurar a propósito (`adapters.py:120-124`, y el filtro de log
   `_DropUnconfiguredRoleLog` en `lightrag_store.py:38-62`).
2. Llama a `rag.aquery_llm(query, param)` (no `aquery`, que descarta el
   `raw_data`) — `lightrag_store.py:249, 272`.
3. `_wrap_query_response` (`lightrag_store.py:322`) mete el string de contexto en
   `Document.page_content` y el grafo estructurado en
   `metadata["lightrag_raw_data"]`.
4. `_normalize_lightrag_graph` (`lightrag_store.py:377`) traduce la forma de
   LightRAG (`entity_name`/`src_id`/…) a la que espera el frontend
   (`id`/`name`/`source`/`target`).
5. `_extract_lightrag_keywords` (`lightrag_store.py:287`) limpia del contexto los
   artefactos internos: el JSON de keywords (`high_level_keywords`) y los tags
   `[Data: Reports (N)]`.

**`aquery_llm` — forma de retorno** (`lightrag.py:2387`, verificado en
`1.5.5rc1`):
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
`lightrag.py:2181`): `local`=entidades+chunks, `global`=relaciones+entidades,
`hybrid`=merge round-robin, `mix`=KG+chunks vectoriales, `naive`=solo chunks
vectoriales, `bypass`=sin recuperación.

**Otra ruta** — contexto de grafo directo:
`LightRAGStore.aretrieve_graph_context` (`lightrag_store.py:896`) llama a
`aquery_llm` con `only_need_context=True` y devuelve el grafo normalizado
**sin descartar** cuando el contexto textual está vacío (a diferencia del
retriever). Lo usa `SiloService._search_via_lightrag_retriever`
(`silo_service.py:2166-2205`) para el buscador de silos.

---

## 8. Almacenamiento

Elegido en `build_storage_config` (`storage_config.py:29-33, 127-132`). LightRAG
usa **cuatro** backends simultáneos:

| Rol LightRAG | Backend | Clase LightRAG | Archivo (contenedor, `1.5.5rc1`) |
|---|---|---|---|
| `graph_storage` | **Neo4j** (siempre) | `Neo4JStorage` | `kg/neo4j_impl.py:68` |
| `vector_storage` | **Qdrant** (def.) o **PGVector** | `QdrantVectorDBStorage` / `PGVectorStorage` | `kg/qdrant_impl.py:171` / `kg/postgres_impl.py:3269` |
| `kv_storage` | **PostgreSQL** | `PGKVStorage` | `kg/postgres_impl.py:2535` |
| `doc_status_storage` | **PostgreSQL** | `PGDocStatusStorage` | `kg/postgres_impl.py:4819` |

**Selección del vector store** (`storage_config.py:35-38, 101-107`):
`SUPPORTED_LIGHTRAG_VECTOR_DB_TYPES = {"QDRANT": ..., "PGVECTOR": ...}`; el silo
lo elige con `lightrag_vector_db_type` (def. `QDRANT`,
`lightrag_store.py:507`).

**Variables de entorno exportadas** (`storage_config.py:112-125`, con
`os.environ.setdefault` para no pisar overrides del operador):
- Neo4j: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
- Postgres (parseadas del `SQLALCHEMY_DATABASE_URI` en `_parse_postgres_uri`,
  `storage_config.py:41-60`): `POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE`.
- Qdrant (solo si vector_db=QDRANT): `QDRANT_URL`, `QDRANT_API_KEY`.

**Aislamiento multi-tenant**: cada silo = un `workspace` = `silo_{id}`
(`lightrag_store.py:475, 586`). En Neo4j el workspace es un **label** de nodo;
en Postgres/Qdrant es una **columna/prefijo** `workspace`. Por construcción, no
hay fuga entre silos (`silo_graph_service.py:1-12`).

**Nombres de tablas/colecciones**:
- Postgres (mapa `NAMESPACE_TABLE_MAP`, `kg/postgres_impl.py:7941`), verificado
  en `1.5.5rc1`:
  `LIGHTRAG_DOC_FULL`, `LIGHTRAG_DOC_CHUNKS`, `LIGHTRAG_FULL_ENTITIES`,
  `LIGHTRAG_FULL_RELATIONS`, `LIGHTRAG_ENTITY_CHUNKS`, `LIGHTRAG_RELATION_CHUNKS`,
  `LIGHTRAG_LLM_CACHE`, `LIGHTRAG_VDB_CHUNKS`, `LIGHTRAG_VDB_ENTITY`,
  `LIGHTRAG_VDB_RELATION`, `LIGHTRAG_DOC_STATUS`.
- Qdrant: colecciones con prefijo `lightrag_{workspace}_` (limpieza en
  `_cleanup_qdrant`, `lightrag_store.py:787`).

> 🐞 **BUG CONFIRMADO — `_cleanup_postgres` no limpia nada.** El helper
> `_cleanup_postgres` (`lightrag_store.py:806-813`) borra de tablas
> `kv_store_full_docs`, `kv_store_text_chunks`, `kv_store_full_entities`,
> `kv_store_full_relations`, `kv_store_llm_response_cache` y `doc_status`. Esos
> nombres son los del backend **JSON** de LightRAG (ficheros `kv_store_*.json`),
> **no** los del backend PostgreSQL, cuyas tablas son `LIGHTRAG_*` /
> `LIGHTRAG_DOC_STATUS` (confirmado en `NAMESPACE_TABLE_MAP`,
> `kg/postgres_impl.py:7941`). El `DELETE` va envuelto en `try/except pass`
> (líneas 816-822), así que **falla en silencio** contra tablas inexistentes y
> retiene todas las filas del silo en Postgres al borrar el silo. El mismo
> desajuste afecta a `count_documents` (`lightrag_store.py:958-960`), que
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

LightRAG `1.5.5rc1` enruta el LLM por **rol** (`llm_roles.py`): `extract`,
`keyword`, `query`, `vlm` (`ROLES` en `llm_roles.py`). Cada rol es un
`RoleLLMConfig` (`llm_roles.py:63`) con un `func`; **si un rol no tiene `func`,
cae al `llm_model_func` base** (`raw_func=cfg.func or base_llm_func`,
`lightrag.py:1264`; el base no puede ser `None`, `lightrag.py:1198-1199`).

El repo mapea sus `AIService` a roles en `build_role_llm_configs`
(`adapters.py:106-157`):
- `extract` ← `silo.extract_service` (obligatorio; también es el `func` base).
- `keyword` ← `silo.keywords_service` (opcional; ojo: clave **singular**
  `"keyword"`, no `"keywords"` — `adapters.py:147-149`).
- `vlm` ← `silo.vlm_service` (opcional).
- `query` **se omite a propósito** (siempre `only_need_context=True`, §7).

Cada `func` de rol se crea con `build_llm_model_func` (`adapters.py:160`), que
reusa `create_llm_from_service` (`tools/aiServiceTools.py`) con
`temperature=0.0`. La firma expuesta a LightRAG es
`async def(prompt, system_prompt=None, history_messages=None, **kwargs) -> str`
(`adapters.py:185`). Las llamadas se etiquetan con
`metadata={"lc_source": "lightrag"}` para que el filtro
`_INTERNAL_LC_SOURCES` no filtre el JSON de keywords al stream de chat
(`adapters.py:200-201`).

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
(`lightrag_store.py:621-630`), **independientemente** de la variable de
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
> el *parser* (`_process_json_extraction_result`, `operate.py:722`, que usa
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
— `_handle_entity_relation_summary` (`operate.py:436`) lo reutiliza para
resumir descripciones de entidades en **texto plano** al fusionar. Forzarle el
esquema corrompería todas las descripciones del grafo. LightRAG consume
`_priority` antes de llamar (`utils.py:2080`), así que la llamada no trae
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

`build_embedding_func` (`adapters.py:302`) construye un `EmbeddingFunc`
(`lightrag/utils.py:498`) desde `silo.embedding_service`, reusando
`get_embeddings_model` (`tools/embeddingTools.py`). Detecta la dimensión con una
llamada de sonda a `"."` (`adapters.py:322-323`). `max_token_size` sale de una
tabla por modelo (`_EMBEDDING_MAX_TOKENS_BY_MODEL`, `adapters.py:54-61`; default
`8192`, `adapters.py:52`). Campos de `EmbeddingFunc`: `embedding_dim`, `func`,
`max_token_size`, `model_name` (`utils.py:530-534`).

### 9.3 Métricas de tokens

Durante la indexación, cada llamada LLM/embedding reporta tokens a un
`IndexingTokenAccumulator` (`token_accumulator.py:29`) vía un `contextvar`
(`adapters.py:215-250` para LLM, `:341-344` para embeddings). Prioridad de
fuente: `usage_metadata` de LangChain → `response_metadata.token_usage` →
estimación con tiktoken (`adapters.py:221-248`). Los totales se guardan en la
tabla `indexing_metric` (`models/indexing_metric.py:11`), un row por run
(`prompt_tokens`, `completion_tokens`, `total_tokens`, `tokens_source`,
`llm_calls`, `embedding_tokens`, `duration_seconds`, `cost`, `model_name`).

---

## 10. Cómo lo usa específicamente este repositorio

### 10.1 Wiring general

- **Factory**: `VectorStoreFactory._create_lightrag_backend`
  (`tools/vector_store_factory.py:140-175`) es el único sitio que instancia
  `LightRAGStore`. Comprueba `is_lightrag_available()` y exige
  `ai_service` + `embedding_service`. `IMPLEMENTED_TYPES` incluye `'LIGHTRAG'`
  (`vector_store_factory.py:33`).
- **Un silo LightRAG = un workspace** `silo_{id}`. Se configura por-silo (§3.2)
  desde `SiloService` (`services/silo_service.py`) y `RepositoryService`
  (`services/repository_service.py:68-130`).

### 10.2 Indexación

`SiloService` → `LightRAGStore.index_documents` (§5). Los documentos vienen de
`Repository` (ficheros) o `Domain` (scraping) vectorizados en el silo. Métricas
en `indexing_metric`.

### 10.3 Recuperación en la ejecución del agente

- El agente obtiene un `LightRAGRetriever` vía `get_retriever` /
  `get_retriever_tool` (`tools/agentTools.py:646-674, 1244`). El modo se resuelve
  desde `agent.lightrag_query_mode` (`agentTools.py:661-673`); `skill-routed` es
  un caso especial (`agentTools.py:214, 650`).
- **Citas inline**: `_append_lightrag_citation_sources`
  (`agentTools.py:1044-1071`) numera los chunks de `lightrag_raw_data.data.chunks`
  y pide al LLM citar como `[N](cite://N)`. La numeración es global en el turno
  (patrón de `offset` mutable) para soportar multi-silo.
- **Streaming del grafo**: `agent_streaming_service.py:329-342` acumula los
  eventos `_lightrag_graph` del turno con `merge_lightrag_graph`
  (`tools/streaming_utils.py`) y los emite como `lightrag_graph` en la respuesta
  final (`agent_streaming_service.py:420`) → el frontend los pinta como burbuja
  de grafo (`LightRAGGraphBubble`).

### 10.4 Endpoint de grafo del silo

`GET .../silos/{id}/graph` → `get_silo_graph` (`routers/internal/silos.py:856-884`)
→ `SiloGraphService.get_silo_graph` (`silo_graph_service.py:58`). **Siempre usa
Cypher directo** (no la API `get_knowledge_graph` de LightRAG) porque LightRAG
guarda el workspace como **label**, no como propiedad, y su API filtra por
propiedad → devolvería vacío (`silo_graph_service.py:74-85`). Filtra por label
`silo_{id}` con backticks, cuenta solo nodos con `entity_id` (excluye chunks),
y ordena por grado (`silo_graph_service.py:215-288`).

### 10.5 Manejo del event loop (concurrencia)

Detalle no obvio pero crítico: Neo4j (driver async) liga sus futures a un loop
concreto. El repo usa `_aget_rag_instance` en la ruta de query para inicializar
Neo4j **en el mismo loop** que corre `aquery_llm`, evitando "Future attached to a
different loop" (`lightrag_store.py:605-630, 254-275`). `_run_async`
(`lightrag_store.py:166-200`) reusa el loop del hilo en contexto síncrono y
descarga a un hilo worker si ya hay un loop corriendo.

---

## 11. Checklist de verificación rápida

| Afirmación | Comando / archivo |
|---|---|
| Versión que corre = `1.5.5rc1` | `docker compose run --rm backend python -c "import importlib.metadata as m; print(m.version('lightrag-hku'))"` |
| **Al subir versión**: el JSON guiado sigue activo | Buscar en el log del backend `LightRAG guided JSON`: debe decir `on (marker: ...)`. Un `off:` indica que el marcador ya no se puede derivar (ver 9.1.3) |
| Versión fijada = `1.5.5rc1` | `backend/Dockerfile:75` |
| Servicio = `backend` | `docker/docker-compose.yaml:42` |
| Feature flag | `backend/config.py:63` + `adapters.py:80` |
| Instanciación de LightRAG | `lightrag_store.py:584` |
| Pipeline de inserción | `lightrag_store.py:104-163, 636` |
| Extracción de entidades | `operate.py:3328` (contenedor) |
| Queries (`only_need_context`) | `lightrag_store.py:203-275` |
| Storage backends | `storage_config.py:29-33, 127-132` |
| Bug `_cleanup_postgres` | `lightrag_store.py:806-813` vs `kg/postgres_impl.py:7941` |
| LLM por rol | `adapters.py:106-157` |
| Embeddings | `adapters.py:302-353` |
| Grafo por silo | `silo_graph_service.py:58-85` |

---

*Documento generado analizando el código instalado en el contenedor
`mattin-backend` (`lightrag 1.5.5rc1`) y la capa de integración del repositorio.
Ninguna afirmación de internals se basa en memoria; todas se verificaron sobre
la fuente instalada. Los números de línea del contenedor corresponden a
`1.5.5rc1`; revalidar si se vuelve a subir la versión de `lightrag-hku`.*
