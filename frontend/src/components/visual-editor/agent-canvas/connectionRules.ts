import type { AgentType, PieceKind, PortDef } from './agentPorts';

/**
 * Pure connection rules for the per-agent canvas: whether a piece may be
 * connected to (or disconnected from) a port, with exact user-facing reasons.
 * `port` is passed in instead of calling `portsFor` because pure modules
 * cannot import at runtime.
 */

export interface RuleContext {
  readonly canEdit: boolean;
  readonly agentId: number; // 0 for an unsaved agent
  readonly agentType: AgentType;
  readonly hasMemory: boolean;
  readonly port: PortDef; // target port
  readonly kind: PieceKind;
  readonly pieceId: number;
  readonly lifecycleStatus?: 'active' | 'deprecated' | 'retired';
  /** AI services only: whether the service can analyse video (absent = unknown = no). */
  readonly supportsVideo?: boolean;
  readonly toolEdges: Readonly<Record<number, readonly number[]>>; // from getAgents().tool_ids
}

export type RuleResult = { ok: true; warning?: string } | { ok: false; reason: string };

// With the article, so messages read "An AI service" / "A skill".
export const KIND_PHRASES: Record<PieceKind, string> = {
  ai_service: 'An AI service',
  embedding: 'An embedding service',
  sandbox: 'A sandbox service',
  silo: 'A silo',
  skill: 'A skill',
  mcp: 'An MCP server',
  agent: 'An agent',
  parser: 'An output parser',
  middleware: 'A middleware',
};

const READ_ONLY = "You don't have permission to edit this agent";

/** Connecting `toolId` as a tool of `agentId` creates a cycle if `toolId` transitively uses `agentId`. */
export function wouldCreateToolCycle(
  agentId: number,
  toolId: number,
  toolEdges: Readonly<Record<number, readonly number[]>>,
): boolean {
  const seen = new Set<number>();
  const stack = [toolId];
  while (stack.length > 0) {
    const current = stack.pop() as number;
    if (current === agentId) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    for (const next of toolEdges[current] ?? []) stack.push(next);
  }
  return false;
}

export function canConnect(ctx: RuleContext): RuleResult {
  if (!ctx.canEdit) return { ok: false, reason: READ_ONLY };
  if (!ctx.port.accepts.includes(ctx.kind)) {
    return { ok: false, reason: `${KIND_PHRASES[ctx.kind]} can't be connected to ${ctx.port.label}` };
  }
  if (ctx.port.id === 'video' && ctx.kind === 'ai_service' && ctx.supportsVideo !== true) {
    return { ok: false, reason: "This AI service doesn't support video" };
  }
  if (ctx.kind === 'agent') {
    if (ctx.pieceId === ctx.agentId) return { ok: false, reason: "An agent can't use itself as a tool" };
    if (wouldCreateToolCycle(ctx.agentId, ctx.pieceId, ctx.toolEdges)) {
      return { ok: false, reason: 'This would create a loop: that agent already uses this one as a tool' };
    }
  }
  if (ctx.lifecycleStatus === 'retired') {
    return { ok: false, reason: "This service is retired and can't be connected" };
  }
  if (ctx.lifecycleStatus === 'deprecated') {
    return { ok: true, warning: 'This service is deprecated and will stop working soon' };
  }
  return { ok: true };
}

export function canDisconnect(ctx: Pick<RuleContext, 'canEdit' | 'agentType' | 'hasMemory' | 'port'>): RuleResult {
  if (!ctx.canEdit) return { ok: false, reason: READ_ONLY };
  if (ctx.port.id === 'media_embedding' && ctx.hasMemory) {
    return {
      ok: false,
      reason: 'Conversational agents need a media embedding service. Turn off Conversational first.',
    };
  }
  return { ok: true };
}
