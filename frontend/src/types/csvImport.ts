/**
 * TypeScript interfaces for the CSV → PDF import feature.
 * Mirrors the Pydantic schemas in backend/schemas/import_job_schemas.py.
 */

export type ImportJobStatus = 'DOWNLOADING' | 'REVIEW';

export type ImportRowStatus = 'PENDING' | 'DOWNLOADING' | 'DOWNLOADED' | 'FAILED' | 'CONFIRMED' | 'DISCARDED';

export interface ImportJobRow {
  id: number;
  url: string;
  row_metadata: Record<string, string>;
  status: ImportRowStatus;
  failure_reason: string | null;
  resource_id: number | null;
}

export interface ImportJobCounts {
  total: number;
  pending: number;
  downloading: number;
  downloaded: number;
  failed: number;
  confirmed: number;
  discarded: number;
}

export interface ImportJob {
  id: number;
  repository_id: number;
  status: ImportJobStatus;
  source_filename: string | null;
  link_column: string | null;
  created_at: string | null;
  last_activity_at: string | null;
  rows: ImportJobRow[];
  counts: ImportJobCounts;
}

export interface ConfirmImportRowsResult {
  created_resources: Array<{
    resource_id: number;
    uri: string;
    repository_id: number | null;
    create_date: string | null;
    size: number | null;
    content_type: string;
  }>;
  failed_files: Array<Record<string, unknown>>;
  session_id: string | null;
}
