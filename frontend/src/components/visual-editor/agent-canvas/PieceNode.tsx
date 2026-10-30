import { Handle, Position, type NodeProps } from '@xyflow/react';
import { PIECE_VISUALS } from '../nodeKindConfig';
import type { CanvasFlowNode } from './AgentPortNode';

/**
 * A connected piece (AI service, silo, skill, MCP config, agent tool, ...)
 * around the agent body, with the single source handle new connections are
 * dragged from. Same card look as the old editor's entity cards.
 */
export function PieceNode({ data, isConnectable }: NodeProps<CanvasFlowNode>) {
  const piece = data.piece;
  if (!piece) return null;
  const visual = PIECE_VISUALS[piece.kind];
  const Icon = visual.icon;

  return (
    <div
      className={`group relative flex items-center gap-3 rounded-xl border shadow-sm transition-shadow duration-150 hover:shadow-md ${visual.borderClass} ${visual.bgClass} min-w-[190px] px-3 py-2.5`}
    >
      <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${visual.chipClass}`}>
        <Icon className={`h-4 w-4 ${visual.iconClass}`} aria-hidden="true" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100" title={piece.name}>
          {piece.name}
        </p>
        <p className={`truncate text-xs ${visual.captionClass}`}>{visual.typeCaption}</p>
        {data.portLabels && data.portLabels.length > 0 && (
          <p className="truncate text-[11px] text-gray-500 dark:text-gray-400">
            {data.portLabels.join(', ')}
          </p>
        )}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={isConnectable}
        className="!h-2.5 !w-2.5 !rounded-full !border-2 !border-white dark:!border-gray-900 !bg-gray-400 dark:!bg-gray-500"
      />
    </div>
  );
}
