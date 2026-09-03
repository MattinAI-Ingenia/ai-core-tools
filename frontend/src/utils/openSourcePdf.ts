import { apiService } from '../services/api';

/** Fetch a silo resource's PDF and open it in a new tab at the given page,
 * via the browser's native PDF viewer (`#page=N` fragment) — no PDF.js needed. */
export async function openSourcePdf(appId: number, siloId: number, resourceId: number, page: number): Promise<void> {
  // Open the tab SYNCHRONOUSLY, before the await below — browsers tie the
  // "user activation" that allows window.open() to the click's own call
  // stack, and it's gone by the time an awaited fetch resolves, so opening
  // after the await gets silently popup-blocked (no tab, no visible error).
  const tab = window.open('', '_blank');
  const blob = await apiService.getSiloResourceFile(appId, siloId, resourceId);
  const url = URL.createObjectURL(blob); // server sends media_type="application/pdf"
  if (tab) {
    tab.location.href = `${url}#page=${page}`;
  }
  // Revoking immediately would race the new tab's fetch of the blob URL — give
  // it a moment to load before releasing the memory.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
