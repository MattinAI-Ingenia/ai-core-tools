import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Network, Plus } from 'lucide-react';
import { apiService } from '../services/api';
import { AppGraphCanvas } from '../components/visual-editor/AppGraphCanvas';
import { NODE_KIND_VISUALS } from '../components/visual-editor/nodeKindConfig';
import Modal from '../components/ui/Modal';
import SkillForm from '../components/forms/SkillForm';
import { useApiMutation } from '../hooks/useApiMutation';
import { MESSAGES, errorMessage } from '../constants/messages';
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
  const mutate = useApiMutation();

  const [app, setApp] = useState<App | null>(null);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);

  // AppGraphCanvas owns the actual graph fetch/refetch; this page only
  // needs to trigger one after an action it handles itself (creating a
  // Skill via the inline modal below) - captured in a ref rather than
  // state since calling it is an imperative side effect, not something
  // this component re-renders in response to.
  const refetchGraphRef = useRef<() => Promise<void>>(async () => {});

  const handleRefetchAvailable = useCallback((refetch: () => Promise<void>) => {
    refetchGraphRef.current = refetch;
  }, []);

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

  // Agent and Silo creation reuse their existing full-page forms exactly as
  // the "+ New" buttons on their own list pages do - `returnTo` is the only
  // addition, read by those pages' post-save navigation so finishing sends
  // the user back to this canvas instead of stranding them on the list.
  const handleCreateAgent = useCallback(() => {
    if (!appId) return;
    navigate(`/apps/${appId}/agents/0`, { state: { returnTo: `/apps/${appId}/visual-editor` } });
  }, [appId, navigate]);

  const handleCreateSilo = useCallback(() => {
    if (!appId) return;
    navigate(`/apps/${appId}/silos/new`, { state: { returnTo: `/apps/${appId}/visual-editor` } });
  }, [appId, navigate]);

  // Skill has no dedicated route (only a modal on the Skills list page), so
  // it gets the same modal inline here instead - same `Modal` + `SkillForm`
  // components SkillsPage.tsx already uses.
  async function handleSaveSkill(data: unknown) {
    if (!appId) return;
    const result = await mutate(() => apiService.createSkill(Number.parseInt(appId), data), {
      loading: MESSAGES.CREATING('skill'),
      success: MESSAGES.CREATED('skill'),
      error: (err) => errorMessage(err, MESSAGES.SAVE_FAILED('skill')),
    });
    if (result === undefined) return;
    setIsSkillModalOpen(false);
    void refetchGraphRef.current();
  }

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
            Visual editor
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Explore the resource graph for {app?.name || appId}. Drag nodes to rearrange, collapse an agent to
            hide its satellites, and use each card&apos;s edit button to jump to that resource. Silo, Skill,
            MCP and Agent-as-tool relationships can be edited directly on the canvas.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-600 dark:border-gray-700 dark:bg-gray-800/50 dark:text-gray-400">
        <p className="font-medium text-gray-700 dark:text-gray-300">How connections work</p>
        <p className="mt-1">
          Each card has two connection dots. The <span className="font-medium">left</span> dot accepts a
          connection - it marks this node as something being attached to. The{' '}
          <span className="font-medium">right</span> dot starts one - drag from it to attach this node to
          another. So dragging from an Agent&apos;s right dot to a Silo, Skill, MCP config, or another
          tool-enabled Agent&apos;s left dot attaches that resource to the agent. Only those four relationship
          kinds are editable this way; AI Service, Embedding Service and Output Parser stay read-only. Select
          an edge and press Delete/Backspace to remove a connection.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Add to canvas:</span>
        {(
          [
            { kind: 'agent' as const, label: 'New Agent', onClick: handleCreateAgent },
            { kind: 'silo' as const, label: 'New Silo', onClick: handleCreateSilo },
            { kind: 'skill' as const, label: 'New Skill', onClick: () => setIsSkillModalOpen(true) },
          ]
        ).map(({ kind, label, onClick }) => {
          const visual = NODE_KIND_VISUALS[kind];
          const Icon = visual.icon;
          return (
            <button
              key={kind}
              type="button"
              onClick={onClick}
              aria-label={label}
              title={label}
              className={`relative flex h-11 w-11 items-center justify-center rounded-full border shadow-sm transition-shadow hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-gray-900 ${visual.chipClass} ${visual.borderClass}`}
            >
              <Icon className={`h-5 w-5 ${visual.iconClass}`} aria-hidden="true" />
              <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-white dark:bg-indigo-500">
                <Plus className="h-3 w-3" aria-hidden="true" strokeWidth={3} />
              </span>
            </button>
          );
        })}
      </div>

      <AppGraphCanvas appId={appId} onEditNode={handleEditNode} onRefetchAvailable={handleRefetchAvailable} />

      <Modal
        isOpen={isSkillModalOpen}
        onClose={() => setIsSkillModalOpen(false)}
        title="Create New Skill"
        size="large"
      >
        <SkillForm skill={null} onSubmit={handleSaveSkill} onCancel={() => setIsSkillModalOpen(false)} />
      </Modal>
    </div>
  );
}

export default VisualEditorPage;
