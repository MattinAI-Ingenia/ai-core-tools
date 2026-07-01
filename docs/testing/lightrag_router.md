# LightRAG Router Test Questions

Source document: *Changing Data Sources in the Age of Machine Learning for Official Statistics* — De Boom & Reusens, Statistics Flanders, 2023 (arXiv:2306.04338).

Ingest this PDF into a LightRAG silo and run the questions below against an agent configured with that silo.  The **Expected mode** column is the retrieval mode you would expect LightRAG's router to select for each question; **Expected answer** is a condensed version of what a correct response should contain.  Both columns are sanity references — the router may legitimately pick a different mode and still return a correct answer.

---

## Naive — keyword / verbatim lookup

These questions can be answered by finding a specific span of text with no cross-entity reasoning.

| # | Question | Expected mode | Expected answer |
|---|----------|---------------|-----------------|
| N1 | Who are the authors of this paper? | naive | Cedric De Boom and Michael Reusens, Statistics Flanders, Belgium |
| N2 | At which workshop was this paper presented? | naive | UNECE Machine Learning for Official Statistics Workshop 2023 |
| N3 | What was the monthly cost of Twitter's enterprise API tier introduced in 2023? | naive | More than 40,000 USD per month |
| N4 | Who coined the phrase "data is the new oil" and when? | naive | Clive Humby, 2006 |
| N5 | What version of Twitter's API was launched in 2021? | naive | Twitter API version 2 |

---

## Local — single-entity / definition questions

These questions require understanding one concept or entity from the knowledge graph.

| # | Question | Expected mode | Expected answer |
|---|----------|---------------|-----------------|
| L1 | What is concept drift and why does it affect machine learning models used for official statistics? | local | Concept drift is a change in data distribution between train and inference time; it causes model deterioration or loss of accuracy, and sociological/economic processes are naturally prone to it. |
| L2 | What is operationalization bias? | local | Reproducibility issues caused by implicit, hidden, or production-specific design choices that affect how a concept is measured or operationalized. |
| L3 | What does "model staleness" mean in this paper? | local | A model that is not updated frequently enough no longer reflects current patterns in the data, leading to less accurate official statistics. |
| L4 | What is a "breaking change" in the context of changing data sources? | local | When a new data source is introduced that mismatches the original, making the resulting statistic incomparable; requires transparency (e.g. marking the change on a graph). |
| L5 | What are the characteristics of external data sources that make them appealing for official statistics? | local | Broad-spectrum coverage, diversity, availability, large size (sometimes complete datasets), varied structure (text/image/video/audio), timeliness, fine-grained frequency, granularity, and geographic coverage. |
| L6 | What are the main challenges of working with external data sources before using them for machine learning? | local | Data quality, data interpretation, data integration, selection bias, operationalization bias, computational resources, privacy/security, data ethics, fairness, and cost. |
| L7 | How does the paper distinguish between training and inference in a machine learning pipeline? | local | During training, model parameters are tuned on data; during inference, parameters are fixed and the model predicts on new data. The distinction matters because a model remains unchanged after training until retrained, causing issues when inference-time data diverges from training data. |

---

## Global — thematic / cross-cutting questions

These questions require synthesizing information across multiple sections of the document.

| # | Question | Expected mode | Expected answer |
|---|----------|---------------|-----------------|
| G1 | What is the central argument of this paper? | global | Statistical agencies using external data sources for ML-driven official statistics face significant risks because they have limited control over those sources; the paper catalogues causes of data source changes, their consequences, and mitigation strategies. |
| G2 | What is the overall conclusion and recommendation for statistical agencies? | global | The risks are numerous and mitigations are costly; agencies should minimize loss of control over data sources, perform early risk analysis, and plan with a multi-year horizon — while acknowledging the genuine opportunities ML offers. |
| G3 | How does the paper use the oil analogy to explain the vulnerability of relying on external data? | global | Just as global economies were vulnerable to oil price and supply fluctuations beyond their control, statistical agencies that depend on external data sources are similarly powerless when those sources change, are discontinued, or become expensive. |
| G4 | What categories of risk does the paper identify when data sources change? | global | Technical (data types/schemas, APIs, concept drift, frequency interruptions), organizational (ownership/discontinuation), legal (GDPR, SLAs, cost), and ethical/public perception risks. |
| G5 | How does the paper characterize the relationship between machine learning model outputs and downstream pipelines? | global | ML model predictions can themselves become data sources for other models; when input data changes, prediction shift propagates through the pipeline like a domino effect, making the entire chain vulnerable to upstream changes. |

---

## Hybrid — multi-hop / mixed specificity

These questions combine entity-level details with thematic reasoning.

| # | Question | Expected mode | Expected answer |
|---|----------|---------------|-----------------|
| H1 | Walk through the Twitter API case study: what changes occurred, and what were their consequences for statistical pipelines? | hybrid | Twitter launched v2 in 2021 (breaking endpoint/field/pricing changes), maintained v1.1 in parallel causing inaction, then Musk acquired Twitter in 2022, suspended all existing API offerings, and introduced an enterprise tier at >$40k/month in 2023 — abandoning many research/statistics initiatives. |
| H2 | What does the paper say about monitoring unsupervised models specifically, and why is it harder than monitoring supervised ones? | hybrid | Unsupervised models lack a direct performance metric; instead, monitor cluster similarity/homogeneity, visualize latent space projections, test against domain knowledge (e.g. expected similar items), or create proxy supervised tasks from the model's output. |
| H3 | How should a statistical agency handle the situation where a data source is discontinued and an alternative is found? | hybrid | Use the alternative but acknowledge the mismatch: normalize statistical properties as much as possible and be transparent by indicating clearly on graphs or publications when the data source changed (a "breaking change"). Also build redundancy upfront by diversifying sources to avoid single points of failure. |
| H4 | What are the legal mitigation strategies the paper recommends, and how do they relate to the risk of ownership change? | hybrid | Negotiate formal data sharing agreements or SLAs with providers that specify terms, legal responsibilities, and consequences of non-compliance. Ownership changes are a trigger for many other risks (legal, availability, cost), so SLAs should explicitly cover change-of-ownership scenarios. |
| H5 | How do ethical considerations and public perception affect both the choice of data sources and the design of machine learning models? | hybrid | If data sources or variables are deemed controversial or discriminatory, agencies may need to switch sources (requiring model retraining) and replace black-box models with interpretable ones to satisfy demands for transparency and accountability — failure risks public loss of trust in official statistics. |

---

## Mix — exhaustive / comparative questions

These questions ask for exhaustive coverage of a category or explicit comparison between two things, where missing a partial match would be a real failure — the kind of question that justifies paying for local + global + naive all at once.

| # | Question | Expected mode | Expected answer |
|---|----------|---------------|-----------------|
| M1 | Compare how the paper treats supervised versus unsupervised learning in the context of official statistics. | mix | Supervised learning trains on labeled data to predict a known target (e.g. predicting happiness from a Twitter profile); unsupervised learning has no target and instead finds patterns/similarities (e.g. identifying similar citizens or companies). The paper also contrasts how each is monitored: supervised against a reference test set/metric, unsupervised via cluster similarity, latent space visualization, or proxy supervised tasks. |
| M2 | List every specific named example, company, or technology the paper mentions as an illustration of changing data sources (e.g. Twitter, McKinsey), and briefly describe each. | mix | Twitter (API v1.1 → v2 migration, then suspension and paid enterprise tier under Elon Musk); McKinsey (2016 report on companies specializing in buying/selling data); GDPR (new privacy regulation cited as a legal-change driver). |
| M3 | What are ALL the distinct causes of changing data sources discussed in Section 3.1, and how does each one specifically threaten a machine learning pipeline? | mix | Data types/schemas (feature mismatches), sharing/collection technology (broken API integrations), concept drift (distribution shift between train/test), frequency/interruptions (sampling-rate changes altering training distribution), ownership/discontinuation (loss of source), legal properties (compliance-driven access changes), ethics/public perception (forced model or source replacement). |
| M4 | Compare the mitigation strategies proposed in Section 3.3 (risk analysis, monitoring, diversification, technical robustness, legal robustness) — which ones address technical risks versus organizational/legal risks? | mix | Technical: monitoring, diversification, technical robustness (unit/integration testing, failover, data normalization). Organizational/legal: risk analysis (upfront due diligence) and legal robustness (SLAs, data sharing agreements, compliance terms). |

---

## Out-of-scope / negative cases

The agent should acknowledge it cannot find the answer in the knowledge base rather than hallucinating.

| # | Question | What to check |
|---|----------|---------------|
| OOS1 | What specific machine learning algorithms (e.g. XGBoost, BERT) does the paper recommend for nowcasting? | Should say none are recommended — the paper describes ML at a high level only. |
| OOS2 | What are the population and GDP figures for Belgium mentioned in this paper? | Not in the paper — agent should not invent statistics. |
| OOS3 | What are Statistics Flanders' annual budget and headcount? | Not covered — agent should defer. |

---

## How to run

1. Upload the PDF to a silo backed by a LightRAG store.
2. Create an agent linked to that silo.
3. Ask each question in the playground and record:
   - Whether the subgraph bubble appears (entities/chunks count).
   - Whether the answer is factually correct against the expected answer above.
   - Which retrieval mode LightRAG selected (visible in logs or the `query_mode` metadata on the subgraph).
4. For OOS questions, verify the agent does **not** confabulate an answer.
