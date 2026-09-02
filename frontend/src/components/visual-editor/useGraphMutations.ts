import { useCallback, useState } from 'react';
import { toast } from 'sonner';
import { MESSAGES, errorMessage } from '../../constants/messages';
import { apiService, type Agent } from '../../services/api';
import { parseNodeId, type GraphNode } from '../../hooks/useAppGraph';
import { buildRelationshipChange } from './agentRelationshipMutation';
import { toFlowEdges, type AppFlowEdge } from './graphAdapter';
import { resolveConnection, type ResolvedConnection } from './graphConnectionRules';

export interface UseGraphMutationsOptions {
  readonly appId: string | number | undefined;
  readonly graphNodes: readonly GraphNode[];
  /** Called after a successful mutation - typically `useAppGraph`'s `refetch`. */
  readonly onSettled: () => Promise<void>;
}

export interface UseGraphMutationsResult {
  /** Optimistically-added edges not yet reflected in the fetched graph. */
  readonly pendingEdges: readonly AppFlowEdge[];
  /** Edge ids optimistically hidden pending a delete mutation. */
  readonly hiddenEdgeIds: ReadonlySet<string>;
  readonly connect: (connection: { source: string | null; target: string | null }) => void;
  /**
   * Removes a batch of edges (e.g. from a multi-select delete). Edges owned
   * by the same agent are folded into a single fetch-then-update round trip
   * so concurrent removals on the same agent never race each other; distinct
   * agents' groups are mutated independently and a failure in one never
   * blocks the others.
   */
  readonly disconnectMany: (edges: readonly AppFlowEdge[]) => void;
}

function parseAppId(appId: string | number | undefined): number | null {
  if (appId === undefined || appId === '') return null;
  const numeric = typeof appId === 'number' ? appId : Number.parseInt(appId, 10);
  return Number.isNaN(numeric) ? null : numeric;
}

async function applyRelationshipMutation(
  numericAppId: number,
  resolved: ResolvedConnection,
  mode: 'add' | 'remove',
  onSettled: () => Promise<void>,
): Promise<void> {
  try {
    const agent = await apiService.getAgent(numericAppId, resolved.agentNumericId);
    const change = buildRelationshipChange(
      agent,
      { kind: resolved.kind, targetNumericId: resolved.targetNumericId },
      mode,
    );
    await apiService.updateAgent(numericAppId, resolved.agentNumericId, { ...agent, ...change });
    toast.success(MESSAGES.UPDATED('connection'));
    await onSettled();
  } catch (err) {
    toast.error(errorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
  }
}

/**
 * Removes every edge in `groupEdges` (all owned by the same `agentNumericId`)
 * with a single fetch + single update round trip, folding each removal's
 * patch into the next so multiple removals on the same relationship array
 * (e.g. two skill edges deleted together) compose correctly instead of one
 * overwriting the other.
 */
async function disconnectAgentGroup(
  numericAppId: number,
  agentNumericId: number,
  groupEdges: readonly AppFlowEdge[],
): Promise<void> {
  const agent = await apiService.getAgent(numericAppId, agentNumericId);
  let workingAgent: Omit<Agent, 'silo_id'> & { silo_id?: number | null } = agent;
  let patch: Omit<Partial<Agent>, 'silo_id'> & { silo_id?: number | null } = {};

  for (const edge of groupEdges) {
    const kind = edge.data?.kind;
    if (!kind) continue;
    const change = buildRelationshipChange(
      workingAgent as Agent,
      { kind, targetNumericId: parseNodeId(edge.target).numericId },
      'remove',
    );
    patch = { ...patch, ...change };
    workingAgent = { ...workingAgent, ...change };
  }

  await apiService.updateAgent(numericAppId, agentNumericId, { ...agent, ...patch });
}

export function useGraphMutations({
  appId,
  graphNodes,
  onSettled,
}: UseGraphMutationsOptions): UseGraphMutationsResult {
  const [pendingEdges, setPendingEdges] = useState<readonly AppFlowEdge[]>([]);
  const [hiddenEdgeIds, setHiddenEdgeIds] = useState<ReadonlySet<string>>(new Set());

  const connect = useCallback(
    (connection: { source: string | null; target: string | null }) => {
      const numericAppId = parseAppId(appId);
      if (numericAppId === null) return;

      const resolved = resolveConnection(graphNodes, connection);
      if (!resolved) {
        toast.error("This relationship can't be edited from the canvas");
        return;
      }

      const optimisticEdge = toFlowEdges([
        {
          id: `pending:${connection.source}->${connection.target}`,
          source: connection.source as string,
          target: connection.target as string,
          kind: resolved.kind,
        },
      ])[0];

      setPendingEdges((current) => [...current, optimisticEdge]);
      applyRelationshipMutation(numericAppId, resolved, 'add', onSettled).finally(() => {
        setPendingEdges((current) => current.filter((edge) => edge.id !== optimisticEdge.id));
      });
    },
    [appId, graphNodes, onSettled],
  );

  const disconnectMany = useCallback(
    (edges: readonly AppFlowEdge[]) => {
      const numericAppId = parseAppId(appId);
      if (numericAppId === null || edges.length === 0) return;

      const groups = new Map<number, AppFlowEdge[]>();
      for (const edge of edges) {
        if (!edge.data?.kind) continue;
        const agentNumericId = parseNodeId(edge.source).numericId;
        const group = groups.get(agentNumericId);
        if (group) {
          group.push(edge);
        } else {
          groups.set(agentNumericId, [edge]);
        }
      }

      const edgeIds = edges.map((edge) => edge.id);
      setHiddenEdgeIds((current) => {
        const next = new Set(current);
        for (const id of edgeIds) next.add(id);
        return next;
      });

      const groupSettlements = Array.from(groups.entries()).map(([agentNumericId, groupEdges]) =>
        disconnectAgentGroup(numericAppId, agentNumericId, groupEdges).catch((err: unknown) => {
          toast.error(errorMessage(err, MESSAGES.UPDATE_FAILED('connection')));
          throw err;
        }),
      );

      Promise.allSettled(groupSettlements)
        .then(async (results) => {
          const succeeded = results.some((result) => result.status === 'fulfilled');
          if (succeeded) {
            toast.success(MESSAGES.UPDATED('connection'));
            await onSettled();
          }
        })
        .finally(() => {
          setHiddenEdgeIds((current) => {
            const next = new Set(current);
            for (const id of edgeIds) next.delete(id);
            return next;
          });
        });
    },
    [appId, onSettled],
  );

  return { pendingEdges, hiddenEdgeIds, connect, disconnectMany };
}
