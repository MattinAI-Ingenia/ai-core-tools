// Agent configuration constants
export const DEFAULT_AGENT_TEMPERATURE = 0.7;

export const DEFAULT_MEMORY_SUMMARIZE_THRESHOLD = 20;

// Retrieval configuration defaults (must stay in sync with the backend
// Agent model defaults in backend/models/agent.py)
export const DEFAULT_RETRIEVAL_SEARCH_TYPE = 'similarity';
export const DEFAULT_RETRIEVAL_K = 30;
export const DEFAULT_RETRIEVAL_STRATEGY = 'passthrough';