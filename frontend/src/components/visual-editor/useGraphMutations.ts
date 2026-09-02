import { useCallback, useState } from 'react';
import { toast } from 'sonner';
import { MESSAGES, errorMessage } from '../../constants/messages';
import { apiService } from '../../services/api';
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
  readonly disconnect: (edge: AppFlowEdge) => void;
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
      if (!resolved) return;

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

  const disconnect = useCallback(
    (edge: AppFlowEdge) => {
      const numericAppId = parseAppId(appId);
      const kind = edge.data?.kind;
      if (numericAppId === null || !kind) return;

      const resolved: ResolvedConnection = {
        kind,
        agentNumericId: parseNodeId(edge.source).numericId,
        targetNumericId: parseNodeId(edge.target).numericId,
      };

      setHiddenEdgeIds((current) => new Set(current).add(edge.id));
      applyRelationshipMutation(numericAppId, resolved, 'remove', onSettled).finally(() => {
        setHiddenEdgeIds((current) => {
          const next = new Set(current);
          next.delete(edge.id);
          return next;
        });
      });
    },
    [appId, onSettled],
  );

  return { pendingEdges, hiddenEdgeIds, connect, disconnect };
}
