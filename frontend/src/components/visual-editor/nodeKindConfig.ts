import {
  Bot,
  Cpu,
  Database,
  Layers,
  Sparkles,
  Plug,
  FileJson,
  type LucideIcon,
} from 'lucide-react';
import type { GraphNodeKind } from '../../hooks/useAppGraph';

/**
 * Visual configuration for a single graph node kind: icon, Tailwind accent
 * classes (light + dark aware) and a human-readable type caption shown under
 * the entity label. Kept as a plain data map (no JSX) so it stays cheap to
 * import from both the node components and the flow adapter.
 */
export interface NodeKindVisual {
  readonly icon: LucideIcon;
  readonly typeCaption: string;
  /** Icon glyph color - intentionally decoupled from `captionClass` because
   *  some kind hues (amber/cyan) need a darker shade for the smaller type
   *  caption text to hit 4.5:1 contrast than looks right for a 16px icon. */
  readonly iconClass: string;
  /** Type caption text color - AA-safe (>=4.5:1) against the card background in both themes. */
  readonly captionClass: string;
  /** Icon chip background. */
  readonly chipClass: string;
  /** Card border color (also used to tint the connected edges). */
  readonly borderClass: string;
  /** Card background. */
  readonly bgClass: string;
  /** Raw hex used for React Flow edge/marker strokes (must match borderClass hue). */
  readonly accentHex: string;
  /** Agents are the primary/central entity - rendered slightly larger and bolder. */
  readonly emphasize: boolean;
}

export const NODE_KIND_VISUALS: Record<GraphNodeKind, NodeKindVisual> = {
  agent: {
    icon: Bot,
    typeCaption: 'Agent',
    iconClass: 'text-indigo-600 dark:text-indigo-400',
    captionClass: 'text-indigo-600 dark:text-indigo-400',
    chipClass: 'bg-indigo-100 dark:bg-indigo-900/40',
    borderClass: 'border-indigo-300 dark:border-indigo-600',
    bgClass: 'bg-gradient-to-br from-indigo-50 to-white dark:from-indigo-950/50 dark:to-gray-900',
    accentHex: '#6366f1',
    emphasize: true,
  },
  service: {
    icon: Cpu,
    typeCaption: 'AI Service',
    iconClass: 'text-blue-600 dark:text-blue-400',
    captionClass: 'text-blue-600 dark:text-blue-400',
    chipClass: 'bg-blue-100 dark:bg-blue-900/40',
    borderClass: 'border-blue-200 dark:border-blue-800',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#2563eb',
    emphasize: false,
  },
  silo: {
    icon: Database,
    typeCaption: 'Silo',
    iconClass: 'text-amber-600 dark:text-amber-400',
    captionClass: 'text-amber-700 dark:text-amber-400',
    chipClass: 'bg-amber-100 dark:bg-amber-900/40',
    borderClass: 'border-amber-200 dark:border-amber-800',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#d97706',
    emphasize: false,
  },
  embedding: {
    icon: Layers,
    typeCaption: 'Embedding',
    iconClass: 'text-cyan-600 dark:text-cyan-400',
    captionClass: 'text-cyan-700 dark:text-cyan-400',
    chipClass: 'bg-cyan-100 dark:bg-cyan-900/40',
    borderClass: 'border-cyan-200 dark:border-cyan-800',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#0891b2',
    emphasize: false,
  },
  skill: {
    icon: Sparkles,
    typeCaption: 'Skill',
    iconClass: 'text-purple-600 dark:text-purple-400',
    captionClass: 'text-purple-600 dark:text-purple-400',
    chipClass: 'bg-purple-100 dark:bg-purple-900/40',
    borderClass: 'border-purple-200 dark:border-purple-800',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#9333ea',
    emphasize: false,
  },
  mcp: {
    icon: Plug,
    typeCaption: 'MCP Config',
    iconClass: 'text-rose-600 dark:text-rose-400',
    captionClass: 'text-rose-600 dark:text-rose-400',
    chipClass: 'bg-rose-100 dark:bg-rose-900/40',
    borderClass: 'border-rose-200 dark:border-rose-800',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#e11d48',
    emphasize: false,
  },
  parser: {
    icon: FileJson,
    typeCaption: 'Output Parser',
    iconClass: 'text-slate-600 dark:text-slate-400',
    captionClass: 'text-slate-600 dark:text-slate-400',
    chipClass: 'bg-slate-100 dark:bg-slate-800',
    borderClass: 'border-slate-200 dark:border-slate-700',
    bgClass: 'bg-white dark:bg-gray-800',
    accentHex: '#475569',
    emphasize: false,
  },
};
