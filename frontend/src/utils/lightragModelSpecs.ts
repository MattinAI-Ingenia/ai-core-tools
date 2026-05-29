// LightRAG 2026.05 role-specific model recommendations.
//
// Each LightRAG role has different requirements:
//   EXTRACT  — entity/relationship extraction. Mid-tier non-reasoning.
//   QUERY    — final answer generation. Large, optionally reasoning.
//   KEYWORDS — query keyword extraction. Small, fast. Only context matters.
//   VLM      — vision-language for images. MUST be multimodal (blocking).
//
// The catalog below is a conservative public estimate per model — when
// uncertain the numbers bias toward the lower bound so warnings err on
// the safe side. The backend mirrors these specs in
// ``backend/services/silo_service.py`` (_MODEL_SPECS).

export type LightRAGRole = 'extract' | 'query' | 'keywords' | 'vlm';

export interface ModelSpecs {
  context_kb: number;
  params_b: number;
  supports_vision: boolean;
}

export interface RoleMinSpec {
  params_b: number | null;
  context_kb: number | null;
}

export const ROLE_MIN_SPECS: Record<LightRAGRole, RoleMinSpec> = {
  extract:  { params_b: 12, context_kb: 32 },
  query:    { params_b: 32, context_kb: 32 },
  keywords: { params_b: null, context_kb: 8 },
  vlm:      { params_b: null, context_kb: null }, // vision flag handled separately
};

const MODEL_SPECS: Record<string, ModelSpecs> = {
  // OpenAI
  'gpt-4o':            { context_kb: 128,  params_b: 200,  supports_vision: true },
  'gpt-4o-mini':       { context_kb: 128,  params_b: 15,   supports_vision: true },
  'gpt-4-turbo':       { context_kb: 128,  params_b: 175,  supports_vision: true },
  'gpt-4':             { context_kb: 8,    params_b: 175,  supports_vision: false },
  'gpt-3.5-turbo':     { context_kb: 16,   params_b: 175,  supports_vision: false },
  // Anthropic
  'claude-opus-4-7':   { context_kb: 200,  params_b: 175,  supports_vision: true },
  'claude-sonnet-4-6': { context_kb: 200,  params_b: 100,  supports_vision: true },
  'claude-opus':       { context_kb: 200,  params_b: 175,  supports_vision: true },
  'claude-3.5-sonnet': { context_kb: 200,  params_b: 100,  supports_vision: true },
  'claude-3-haiku':    { context_kb: 200,  params_b: 30,   supports_vision: true },
  // Mistral
  'mistral-large':     { context_kb: 128,  params_b: 123,  supports_vision: false },
  'mistral-medium':    { context_kb: 32,   params_b: 60,   supports_vision: false },
  'mistral-small':     { context_kb: 32,   params_b: 10,   supports_vision: false },
  'pixtral':           { context_kb: 128,  params_b: 12,   supports_vision: true },
  // Google
  'gemini-2.0-flash':  { context_kb: 1000, params_b: 1000, supports_vision: true },
  'gemini-1.5-pro':    { context_kb: 1000, params_b: 1000, supports_vision: true },
  'gemini-1.5-flash':  { context_kb: 1000, params_b: 1000, supports_vision: true },
};

/** Return known specs for a model name, or null if unknown. */
export function lookupModelSpecs(modelName: string | null | undefined): ModelSpecs | null {
  if (!modelName) return null;
  const lower = modelName.toLowerCase();
  for (const [id, specs] of Object.entries(MODEL_SPECS)) {
    if (lower.includes(id) || id.includes(lower)) return specs;
  }
  return null;
}

/** True only when the model is known to handle image inputs. */
export function modelSupportsVision(modelName: string | null | undefined): boolean {
  const specs = lookupModelSpecs(modelName);
  return Boolean(specs?.supports_vision);
}

/**
 * Return a non-blocking warning for a (role, model) pair, or null if OK.
 * VLM validation is intentionally NOT a warning here — it's a blocking error
 * surfaced separately (see ``vlmBlockingError``).
 */
export function getRoleWarning(role: LightRAGRole, modelName: string | null | undefined): string | null {
  if (!modelName || role === 'vlm') return null;
  const min = ROLE_MIN_SPECS[role];
  const specs = lookupModelSpecs(modelName);

  if (!specs) {
    // Unknown model — fall back to a name heuristic for the strict roles.
    const lower = modelName.toLowerCase();
    if ((role === 'extract' || role === 'query') && ['mini', 'small', 'tiny'].some(p => lower.includes(p))) {
      const ctx = min.context_kb ?? '?';
      const par = min.params_b ?? '?';
      return `'${modelName}' may be too small for ${role.toUpperCase()} (recommend ${par}B+ params, ${ctx}K+ context)`;
    }
    return null;
  }

  const issues: string[] = [];
  if (min.context_kb !== null && specs.context_kb < min.context_kb) {
    issues.push(`context ${specs.context_kb}K < ${min.context_kb}K`);
  }
  if (min.params_b !== null && specs.params_b < min.params_b) {
    issues.push(`params ~${specs.params_b}B < ${min.params_b}B`);
  }
  if (issues.length === 0) return null;
  return `${role.toUpperCase()}: '${modelName}' below recommendation (${issues.join(', ')})`;
}

/**
 * Return a blocking error message when the VLM service isn't multimodal,
 * or null when the slot is empty (VLM is optional) or the model supports vision.
 */
export function vlmBlockingError(modelName: string | null | undefined): string | null {
  if (!modelName) return null; // empty is allowed
  if (modelSupportsVision(modelName)) return null;
  return (
    `VLM role requires a multimodal model — '${modelName}' is not known to support vision. ` +
    `Pick a vision-capable model (gpt-4o, claude-3.5-sonnet, gemini-1.5-pro) ` +
    `or leave the VLM service empty.`
  );
}
