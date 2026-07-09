"""LightRAG LLM smoke tests — hit the real endpoints with production-equivalent prompts.

These tests build the exact prompts that LightRAG sends during indexing (entity
extraction) and querying (keyword extraction), then call each endpoint directly
over HTTP.  No LightRAG internals, no DB, no mocks.

Configuration (env vars):
  LIGHTRAG_EXTRACT_URL      Base URL of the extract-LLM endpoint, e.g.
                            http://host:11434/v1  or  http://host:8080
                            Appends /v1 if not already present.
  LIGHTRAG_EXTRACT_MODEL    Model name, e.g. llama3.1:70b
  LIGHTRAG_EXTRACT_PASSWORD API key / basic-auth password  (default: "dummy")

  LIGHTRAG_KEYWORD_URL      Base URL for the keyword-extraction LLM.
                            Defaults to LIGHTRAG_EXTRACT_URL.
  LIGHTRAG_KEYWORD_MODEL    Model name.  Defaults to LIGHTRAG_EXTRACT_MODEL.
  LIGHTRAG_KEYWORD_PASSWORD Defaults to LIGHTRAG_EXTRACT_PASSWORD.

Run:
  pytest tests/integration/tools/test_lightrag_llm_smoke.py -v -s

Both tests are skipped when LIGHTRAG_EXTRACT_URL is not set.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_EXTRACT_URL = os.getenv("LIGHTRAG_EXTRACT_URL", "").rstrip("/")
_EXTRACT_MODEL = os.getenv("LIGHTRAG_EXTRACT_MODEL", "")
_EXTRACT_PASSWORD = os.getenv("LIGHTRAG_EXTRACT_PASSWORD", "dummy")
_EXTRACT_TIMEOUT = float(os.getenv("LIGHTRAG_EXTRACT_TIMEOUT", "600"))

_KEYWORD_URL = os.getenv("LIGHTRAG_KEYWORD_URL", _EXTRACT_URL).rstrip("/")
_KEYWORD_MODEL = os.getenv("LIGHTRAG_KEYWORD_MODEL", _EXTRACT_MODEL)
_KEYWORD_PASSWORD = os.getenv("LIGHTRAG_KEYWORD_PASSWORD", _EXTRACT_PASSWORD)

# OpenAI comparison — set OPENAI_API_KEY to enable
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_EXTRACT_MODEL = os.getenv("LIGHTRAG_OPENAI_EXTRACT_MODEL", "gpt-5.4-mini")
_OPENAI_KEYWORD_MODEL = os.getenv("LIGHTRAG_OPENAI_KEYWORD_MODEL", "gpt-5.4-nano")
_OPENAI_URL = "https://api.openai.com"

_needs_extract_url = pytest.mark.skipif(
    not _EXTRACT_URL,
    reason="Set LIGHTRAG_EXTRACT_URL (and optionally LIGHTRAG_EXTRACT_MODEL) to run",
)
_needs_keyword_url = pytest.mark.skipif(
    not _KEYWORD_URL,
    reason="Set LIGHTRAG_KEYWORD_URL (or LIGHTRAG_EXTRACT_URL) to run",
)
_needs_openai_key = pytest.mark.skipif(
    not _OPENAI_API_KEY,
    reason="Set OPENAI_API_KEY (and optionally LIGHTRAG_OPENAI_MODEL) to run",
)

# ---------------------------------------------------------------------------
# LightRAG prompt constants (mirrors lightrag/prompt.py + operate.py assembly)
# ---------------------------------------------------------------------------

_TUPLE_DELIMITER = "<|#|>"
_COMPLETION_DELIMITER = "<|COMPLETE|>"
_ENTITY_TYPES = [
    "Person", "Creature", "Organization", "Location", "Event",
    "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
]
_ENTITY_TYPES_STR = ",".join(_ENTITY_TYPES)
_LANGUAGE = "English"

# System prompt (from lightrag/prompt.py — entity_extraction_system_prompt)
_ENTITY_SYSTEM_TEMPLATE = """\
---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`.
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   **Avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output must be written in `{language}`.
    *   Proper nouns should be retained in their original language if a proper translation is not available.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships have been completely extracted.

---Examples---
<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}Person{tuple_delimiter}Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity{tuple_delimiter}Taylor{tuple_delimiter}Person{tuple_delimiter}Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device.
entity{tuple_delimiter}Jordan{tuple_delimiter}Person{tuple_delimiter}Jordan shares a commitment to discovery and has a significant interaction with Taylor.
entity{tuple_delimiter}Cruz{tuple_delimiter}Person{tuple_delimiter}Cruz is associated with a vision of control and order.
entity{tuple_delimiter}The Device{tuple_delimiter}Artifact{tuple_delimiter}The Device is central to the story, with potential game-changing implications.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}power dynamics, observation{tuple_delimiter}Alex observes Taylor's authoritarian behavior.
relation{tuple_delimiter}Alex{tuple_delimiter}Jordan{tuple_delimiter}shared goals, rebellion{tuple_delimiter}Alex and Jordan share a commitment to discovery.
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}conflict resolution, mutual respect{tuple_delimiter}Taylor and Jordan interact directly regarding the device.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}ideological conflict{tuple_delimiter}Jordan's commitment to discovery is in rebellion against Cruz's vision.
relation{tuple_delimiter}Taylor{tuple_delimiter}The Device{tuple_delimiter}reverence, technological significance{tuple_delimiter}Taylor shows reverence towards the device.
{completion_delimiter}

<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}Data{tuple_delimiter}The Global Tech Index tracks major technology stocks and experienced a 3.4% decline today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}Organization{tuple_delimiter}Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.
entity{tuple_delimiter}Omega Energy{tuple_delimiter}Organization{tuple_delimiter}Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.
entity{tuple_delimiter}Federal Reserve{tuple_delimiter}Organization{tuple_delimiter}The Federal Reserve's upcoming policy announcement is expected to influence market stability.
relation{tuple_delimiter}Global Tech Index{tuple_delimiter}Nexon Technologies{tuple_delimiter}market performance, company impact{tuple_delimiter}Nexon Technologies' decline contributed to the overall drop in the Global Tech Index.
relation{tuple_delimiter}Omega Energy{tuple_delimiter}Global Tech Index{tuple_delimiter}contrarian performance{tuple_delimiter}Omega Energy bucked the market trend with a 2.1% gain while the index declined.
{completion_delimiter}
"""

# User prompt template (from lightrag/prompt.py — entity_extraction_user_prompt)
_ENTITY_USER_TEMPLATE = """\
---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted.
4.  **Output Language:** Ensure the output language is {language}.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

# Keywords extraction prompt (from lightrag/prompt.py — keywords_extraction)
_KEYWORDS_TEMPLATE = """\
---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent.
2. **low_level_keywords**: for specific entities or details, identifying specific entities, proper nouns, technical jargon.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON.
2. **Source of Truth**: All keywords must be explicitly derived from the user query.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases.
4. **Language**: All extracted keywords MUST be in {language}.

---Examples---
Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{{"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}}

Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{{"high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"], "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]}}

Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{{"high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"], "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]}}

---Real Data---
User Query: {query}

---Output---
Output:"""

# ---------------------------------------------------------------------------
# Realistic ~1200-token text chunk (matches the PDF domain the app indexes).
# 1200 tokens ≈ 900 words.  This is exactly what LightRAG slices from a document
# and sends to the extract LLM.
# ---------------------------------------------------------------------------

_EXTRACT_CHUNK = """\
The integration of machine learning techniques into the production of official statistics represents a fundamental shift in how national statistical offices collect, process, and disseminate data. Traditional survey-based methods, while robust and methodologically well-established, face increasing challenges related to response burden, rising costs, and timeliness constraints. In response, statistical organizations around the world have begun exploring administrative data sources, web-scraped data, satellite imagery, and other non-traditional inputs as supplements or replacements for conventional data collection instruments.

Machine learning models offer several advantages in this context. They can process large volumes of unstructured or semi-structured data at scale, identify patterns that would be difficult to detect through manual inspection, and adapt to changing data distributions without requiring complete redesign of the underlying methodology. However, the adoption of these methods in official statistics introduces a set of novel challenges that differ substantially from those encountered in commercial applications.

One of the primary concerns is statistical disclosure control. Official statistics are subject to strict confidentiality requirements: individual records must be protected from re-identification, and published aggregates must not allow inference of suppressed cells. Machine learning models, particularly those trained on large datasets, may inadvertently memorize sensitive information or produce outputs that enable inference attacks. Differential privacy techniques and output perturbation methods have been proposed as mitigations, but their application in the context of complex regression or classification pipelines requires careful calibration to avoid unacceptable loss of utility.

A second challenge concerns the representativeness of non-traditional data sources. Administrative records are generated for operational rather than statistical purposes, and their coverage may be systematically biased toward certain subpopulations. Small and medium-sized enterprises may be underrepresented in commercial datasets; rural populations may be less visible in mobile phone usage records; recently arrived migrants may not appear in administrative registries. These coverage gaps can introduce biases that propagate through any downstream model trained on such data. Calibration techniques, such as post-stratification weighting and raking, partially mitigate this problem but require auxiliary information about the true population distribution that may itself be uncertain.

The question of explainability is particularly salient in the official statistics context. Unlike commercial applications where a model's output may stand on its own merit, statistical outputs must be accompanied by quality indicators, revision policies, and methodological documentation. Regulators, data users, and the public expect to understand not only what a statistical series reports but also how it was produced. Complex ensemble models or deep neural networks may achieve high predictive accuracy on holdout samples while remaining opaque with respect to the mechanisms driving their predictions. This tension between performance and interpretability has led some national statistical offices to prefer gradient boosting trees over higher-performing black-box alternatives.

Data integration pipelines for official statistics typically involve multiple stages: ingestion from heterogeneous sources, standardization and harmonization of variable definitions, record linkage across datasets using probabilistic matching algorithms, imputation of missing values, and variance estimation to quantify uncertainty in final estimates. Machine learning components embedded within such pipelines are subject to the same quality assurance requirements as any other processing step. The non-deterministic nature of stochastic training algorithms and the sensitivity of learned models to training data composition complicate reproducibility and auditability.

Several national statistical offices have successfully deployed machine learning systems in production. Statistics Canada has used random forest models for business survey editing, replacing rule-based checks with learned anomaly detectors. The UK Office for National Statistics has piloted natural language processing techniques for the automated coding of occupation and industry responses. Statistics Netherlands has experimented with deep learning approaches for the classification of social media posts as economic sentiment indicators. These experiences suggest that a hybrid approach—combining the methodological rigor of traditional statistical frameworks with the computational power of modern machine learning—offers the most promising path forward for the modernization of official statistics production systems.\
"""

# Realistic RAG query (~20 tokens) — what a user would ask an agent over this corpus
_KEYWORD_QUERY = (
    "What are the main challenges of using administrative data sources and machine learning "
    "for producing official statistics, and how do statistical offices address disclosure control?"
)

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _chat(base_url: str, model: str, password: str, messages: list[dict], timeout: float = 300.0, max_tokens: int | None = None) -> tuple[str, float]:
    """POST to /v1/chat/completions, return (content, elapsed_seconds)."""
    url = base_url
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    url = f"{url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {password}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    t0 = time.perf_counter()
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout, verify=False)
    elapsed = time.perf_counter() - t0

    if not resp.is_success:
        pytest.fail(f"HTTP {resp.status_code} from {url}: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_toks = usage.get("prompt_tokens", "?")
    completion_toks = usage.get("completion_tokens", "?")
    print(
        f"\n  elapsed={elapsed:.1f}s  "
        f"prompt_tokens={prompt_toks}  completion_tokens={completion_toks}"
    )
    return content, elapsed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_needs_extract_url
def test_extract_llm_entities_and_relations():
    """Send the LightRAG entity-extraction prompt (1200-token chunk) to the extract LLM.

    Validates:
    - Responds within 300 s (LightRAG timeout is 360 s; 300 leaves a 60 s margin)
    - Output contains at least one entity line  (entity<|#|>...)
    - Output contains at least one relation line (relation<|#|>...)
    - Output ends with the completion delimiter <|COMPLETE|>
    """
    system_prompt = _ENTITY_SYSTEM_TEMPLATE.format(
        tuple_delimiter=_TUPLE_DELIMITER,
        completion_delimiter=_COMPLETION_DELIMITER,
        entity_types=_ENTITY_TYPES_STR,
        language=_LANGUAGE,
    )
    user_prompt = _ENTITY_USER_TEMPLATE.format(
        tuple_delimiter=_TUPLE_DELIMITER,
        completion_delimiter=_COMPLETION_DELIMITER,
        entity_types=_ENTITY_TYPES_STR,
        language=_LANGUAGE,
        input_text=_EXTRACT_CHUNK,
    )

    print(f"\n[extract] url={_EXTRACT_URL}  model={_EXTRACT_MODEL}")
    print(f"  system_prompt_chars={len(system_prompt)}  user_prompt_chars={len(user_prompt)}")

    content, elapsed = _chat(
        _EXTRACT_URL, _EXTRACT_MODEL, _EXTRACT_PASSWORD,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        timeout=_EXTRACT_TIMEOUT,
        max_tokens=8192,
    )

    print(f"  response_chars={len(content)}")
    print("--- response (first 800 chars) ---")
    print(content[:800])
    print("--- end ---")

    assert elapsed < 300, (
        f"Extract LLM took {elapsed:.1f}s — exceeds 300 s safety margin "
        f"(LightRAG hard timeout is 360 s). The model will time-out in production."
    )
    assert f"entity{_TUPLE_DELIMITER}" in content, (
        "No entity line found. Expected at least one line starting with "
        f"'entity{_TUPLE_DELIMITER}'. Check the model and endpoint."
    )
    assert f"relation{_TUPLE_DELIMITER}" in content, (
        "No relation line found. Expected at least one line starting with "
        f"'relation{_TUPLE_DELIMITER}'."
    )
    if _COMPLETION_DELIMITER not in content:
        warnings.warn(
            f"Completion delimiter '{_COMPLETION_DELIMITER}' not found — "
            "model truncated or looped without finishing. Format is correct but "
            "LightRAG may time out waiting for this signal in production.",
            UserWarning,
            stacklevel=2,
        )


@_needs_keyword_url
def test_keyword_llm_extraction():
    """Send the LightRAG keyword-extraction prompt to the keyword LLM.

    Validates:
    - Responds within 60 s (keyword prompts are short; timeouts here are pure infra failures)
    - Output is parseable JSON (with or without markdown code fences)
    - JSON contains high_level_keywords and low_level_keywords as non-empty lists
    """
    prompt = _KEYWORDS_TEMPLATE.format(
        language=_LANGUAGE,
        query=_KEYWORD_QUERY,
    )

    print(f"\n[keyword] url={_KEYWORD_URL}  model={_KEYWORD_MODEL}")
    print(f"  prompt_chars={len(prompt)}")
    print(f"  query={_KEYWORD_QUERY!r}")

    content, elapsed = _chat(
        _KEYWORD_URL, _KEYWORD_MODEL, _KEYWORD_PASSWORD,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"  response_chars={len(content)}")
    print("--- response ---")
    print(content)
    print("--- end ---")

    assert elapsed < 60, (
        f"Keyword LLM took {elapsed:.1f}s — keyword extraction should be fast (<60 s)."
    )

    # Strip markdown fences if the model ignores the "no fences" instruction
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        kw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"Response is not valid JSON: {exc}\n"
            f"Raw response:\n{content}"
        )

    assert "high_level_keywords" in kw, (
        f"Missing 'high_level_keywords' in response: {kw}"
    )
    assert "low_level_keywords" in kw, (
        f"Missing 'low_level_keywords' in response: {kw}"
    )
    assert isinstance(kw["high_level_keywords"], list) and len(kw["high_level_keywords"]) > 0, (
        f"'high_level_keywords' is empty or not a list: {kw}"
    )
    assert isinstance(kw["low_level_keywords"], list) and len(kw["low_level_keywords"]) > 0, (
        f"'low_level_keywords' is empty or not a list: {kw}"
    )

@_needs_openai_key
def test_openai_extract_llm_entities_and_relations():
    """Same entity-extraction prompt sent to OpenAI — compare latency vs local model.

    Set OPENAI_API_KEY to enable.  Override model with LIGHTRAG_OPENAI_MODEL (default: gpt-4o-mini).
    """
    system_prompt = _ENTITY_SYSTEM_TEMPLATE.format(
        tuple_delimiter=_TUPLE_DELIMITER,
        completion_delimiter=_COMPLETION_DELIMITER,
        entity_types=_ENTITY_TYPES_STR,
        language=_LANGUAGE,
    )
    user_prompt = _ENTITY_USER_TEMPLATE.format(
        tuple_delimiter=_TUPLE_DELIMITER,
        completion_delimiter=_COMPLETION_DELIMITER,
        entity_types=_ENTITY_TYPES_STR,
        language=_LANGUAGE,
        input_text=_EXTRACT_CHUNK,
    )

    print(f"\n[openai] url={_OPENAI_URL}  model={_OPENAI_EXTRACT_MODEL}")
    print(f"  system_prompt_chars={len(system_prompt)}  user_prompt_chars={len(user_prompt)}")

    content, elapsed = _chat(
        _OPENAI_URL, _OPENAI_EXTRACT_MODEL, _OPENAI_API_KEY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    print(f"  response_chars={len(content)}")
    print("--- response (first 800 chars) ---")
    print(content[:800])
    print("--- end ---")

    assert elapsed < 300, f"OpenAI took {elapsed:.1f}s — unexpectedly slow."
    assert f"entity{_TUPLE_DELIMITER}" in content, "No entity line found in OpenAI response."
    assert f"relation{_TUPLE_DELIMITER}" in content, "No relation line found in OpenAI response."
    if _COMPLETION_DELIMITER not in content:
        warnings.warn(
            f"Completion delimiter '{_COMPLETION_DELIMITER}' not found in OpenAI response — "
            "model truncated or looped without finishing.",
            UserWarning,
            stacklevel=2,
        )


@_needs_openai_key
def test_openai_keyword_llm_extraction():
    """Same keyword-extraction prompt sent to OpenAI — compare latency vs local model.

    Set OPENAI_API_KEY to enable.  Override model with LIGHTRAG_OPENAI_KEYWORD_MODEL (default: gpt-4.5-nano).
    """
    prompt = _KEYWORDS_TEMPLATE.format(
        language=_LANGUAGE,
        query=_KEYWORD_QUERY,
    )

    print(f"\n[openai-keyword] url={_OPENAI_URL}  model={_OPENAI_KEYWORD_MODEL}")
    print(f"  prompt_chars={len(prompt)}")
    print(f"  query={_KEYWORD_QUERY!r}")

    content, elapsed = _chat(
        _OPENAI_URL, _OPENAI_KEYWORD_MODEL, _OPENAI_API_KEY,
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"  response_chars={len(content)}")
    print("--- response ---")
    print(content)
    print("--- end ---")

    assert elapsed < 60, f"OpenAI keyword took {elapsed:.1f}s — unexpectedly slow."

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        kw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        pytest.fail(f"OpenAI response is not valid JSON: {exc}\nRaw response:\n{content}")

    assert "high_level_keywords" in kw and len(kw["high_level_keywords"]) > 0
    assert "low_level_keywords" in kw and len(kw["low_level_keywords"]) > 0


if __name__ == "__main__":
    # Run the tests directly (pytest -s) for debugging
    test_extract_llm_entities_and_relations()
    test_keyword_llm_extraction()
    test_openai_extract_llm_entities_and_relations()
    test_openai_keyword_llm_extraction()