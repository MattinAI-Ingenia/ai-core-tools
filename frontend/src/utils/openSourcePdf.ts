import { apiService } from '../services/api';

/** Fetch a silo resource's PDF and open it in a new tab at the given page,
 * via the browser's native PDF viewer (`#page=N` fragment) — no PDF.js needed. */
export async function openSourcePdf(appId: number, siloId: number, resourceId: number, page: number): Promise<void> {
  const blob = await apiService.getSiloResourceFile(appId, siloId, resourceId);
  const url = URL.createObjectURL(blob); // server sends media_type="application/pdf"
  window.open(`${url}#page=${page}`, '_blank');
  // Revoking immediately would race the new tab's fetch of the blob URL — give
  // it a moment to load before releasing the memory.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
