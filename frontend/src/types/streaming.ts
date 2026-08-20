export type StreamEventType =
  | 'token'
  | 'tool_start'
  | 'tool_end'
  | 'thinking'
  | 'metadata'
  | 'error'
  | 'done'
  | 'hitl_interrupt';

export interface StreamEvent {
  type: StreamEventType;
  data: Record<string, unknown>;
}

export interface TokenEventData {
  content: string;
}

export interface ToolStartEventData {
  tool_name: string;
  tool_input?: string;
}

export interface ToolEndEventData {
  tool_name: string;
  tool_output?: string;
}

export interface ThinkingEventData {
  message: string;
}

export interface MetadataEventData {
  conversation_id?: number;
  agent_id?: number;
}

export interface ErrorEventData {
  message: string;
}

export interface LightRAGEntity {
  id: string;
  name?: string;
  /** <SEP>-joined chunk ids this entity was extracted from (empty for partial nodes). */
  source_id?: string;
  partial?: boolean;
  [key: string]: unknown;
}

export interface LightRAGRelationship {
  id: string;
  source?: string;
  target?: string;
  /** <SEP>-joined chunk ids this relationship was extracted from. */
  source_id?: string;
  [key: string]: unknown;
}

export interface LightRAGChunk {
  id: string;
  content?: string;
  file_path?: string;
  /** Present only when the chunk's id encodes its source (indexed after this
   * feature shipped) — lets the UI open the source PDF at this page. */
  resource_id?: number;
  page?: number;
  [key: string]: unknown;
}

export interface LightRAGReference {
  reference_id?: string;
  file_path?: string;
  [key: string]: unknown;
}

export interface LightRAGGraphData {
  data?: {
    entities?: LightRAGEntity[];
    relationships?: LightRAGRelationship[];
    chunks?: LightRAGChunk[];
    references?: LightRAGReference[];
  };
}

export interface DoneEventData {
  response: string | Record<string, unknown>;
  files?: Array<{
    file_id: string;
    filename: string;
    file_type: string;
  }>;
  lightrag_graph?: LightRAGGraphData | null;
}

export interface ActiveTool {
  name: string;
  displayName: string;
  input?: string;
  status: 'running' | 'complete';
  startedAt: number;
}

export interface StreamingState {
  isStreaming: boolean;
  content: string;
  activeTools: ActiveTool[];
  thinkingMessage: string | null;
  conversationId: number | null;
  error: string | null;
}
