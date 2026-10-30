/**
 * Pure planning of the saves a connect gesture needs, plus a per-key queue
 * that runs them one at a time without letting one failure block the rest.
 */

import type { PieceKind, PortId } from './agentPorts';

/** One agent save the caller executes: getAgent → updateAgent(appId, id, {...fresh, ...patch}). */
export interface SaveStep {
  readonly agentId: number;
  readonly patch: Record<string, unknown>;
  readonly toast?: string;
}

/** Ordered saves for one connect gesture. */
export function planConnect(args: {
  agentId: number;
  port: PortId;
  kind: PieceKind;
  pieceId: number;
  patch: Record<string, unknown>;
  targetIsTool?: boolean;
  codeInterpreterOn?: boolean;
  replacedName?: string | null;
  pieceName: string;
}): SaveStep[] {
  const steps: SaveStep[] = [];
  if (args.port === 'tools' && args.kind === 'agent' && args.targetIsTool === false) {
    steps.push({
      agentId: args.pieceId,
      patch: { is_tool: true },
      toast: `${args.pieceName} can now be used as a tool`,
    });
  }
  if (args.port === 'sandbox' && args.codeInterpreterOn === false) {
    steps.push({
      agentId: args.agentId,
      patch: { ...args.patch, enable_code_interpreter: true },
      toast: `Code interpreter turned on to use ${args.pieceName}`,
    });
    return steps;
  }
  steps.push({
    agentId: args.agentId,
    patch: args.patch,
    toast: args.replacedName
      ? `Replaced ${args.replacedName} with ${args.pieceName}`
      : `Connected ${args.pieceName}`,
  });
  return steps;
}

/** Runs tasks one at a time per key; a failure rejects that task only and the queue continues. */
export function createSaveQueue(): { run<T>(key: number, task: () => Promise<T>): Promise<T> } {
  const chains = new Map<number, Promise<unknown>>();
  return {
    run<T>(key: number, task: () => Promise<T>): Promise<T> {
      const previous = chains.get(key) ?? Promise.resolve();
      const result = previous.catch(() => {}).then(task);
      chains.set(key, result.catch(() => {}));
      return result;
    },
  };
}
