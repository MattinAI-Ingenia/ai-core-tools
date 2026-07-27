import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Network } from 'lucide-react';
import { apiService } from '../services/api';
import { AppGraphCanvas } from '../components/visual-editor/AppGraphCanvas';
import type { GraphNode } from '../hooks/useAppGraph';
import type { Agent, Silo } from '../services/api';

interface App {
  app_id: number;
  name: string;
}

/**
 * Resolves the editor route for a graph node's underlying entity.
 *
 * Kept here (page-level), never inside `AppGraphCanvas`/`graphAdapter`/
 * `EntityNodeCard` - those stay router-agnostic per the visual editor's
 * design, so only this thin wrapper needs to know about react-router and
 * the app's actual route scheme.
 *
 * `agent` and `silo` have a dedicated per-entity editor route. Every other
 * kind (AI Service, Embedding Service, Skill, MCP Config, Output Parser)
 * is edited inline via a modal on its own list page - there is no
 * dedicated `/:id` edit route for those, so "Edit" falls back to that
 * list page.
 */
function resolveNodeEditPath(appId: string, node: GraphNode): string {
  switch (node.kind) {
    case 'agent': {
      const agent = node.data as Agent;
      return `/apps/${appId}/agents/${agent.agent_id}`;
    }
    case 'silo': {
      const silo = node.data as Silo;
      return `/apps/${appId}/silos/${silo.silo_id}`;
    }
    case 'service':
      return `/apps/${appId}/settings/ai-services`;
    case 'embedding':
      return `/apps/${appId}/settings/embedding-services`;
    case 'skill':
      return `/apps/${appId}/skills`;
    case 'mcp':
      return `/apps/${appId}/settings/mcp-configs`;
    case 'parser':
      return `/apps/${appId}/settings/data-structures`;
    default:
      // Exhaustive above for every `GraphNodeKind` - this is unreachable,
      // but keeps the function total instead of relying on a `never` cast.
      return `/apps/${appId}`;
  }
}

function VisualEditorPage() {
  const { appId } = useParams();
  const navigate = useNavigate();

  const [app, setApp] = useState<App | null>(null);

  useEffect(() => {
    if (!appId) return;
    let cancelled = false;
    apiService
      .getApp(Number.parseInt(appId))
      .then((response) => {
        if (!cancelled) setApp(response);
      })
      .catch((err: unknown) => {
        console.error('Error loading app:', err);
      });
    return () => {
      cancelled = true;
    };
  }, [appId]);

  const handleEditNode = useCallback(
    (node: GraphNode) => {
      if (!appId) return;
      navigate(resolveNodeEditPath(appId, node));
    },
    [appId, navigate],
  );

  if (!appId) {
    return (
      <div className="space-y-6">
        <p className="text-sm text-red-600 dark:text-red-400">Missing app id.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-gray-100">
            <Network className="h-6 w-6 text-indigo-500 dark:text-indigo-400" aria-hidden="true" />
            Editor visual
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Explore the resource graph for {app?.name || appId}. Drag nodes to rearrange, collapse an agent to
            hide its satellites, and use each card&apos;s edit button to jump to that resource.
          </p>
        </div>
      </div>

      <AppGraphCanvas appId={appId} onEditNode={handleEditNode} />
    </div>
  );
}

export default VisualEditorPage;
