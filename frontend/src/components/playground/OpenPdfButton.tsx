import { FileText } from 'lucide-react';
import { openSourcePdf } from '../../utils/openSourcePdf';

interface Props {
  appId?: number;
  siloId?: number;
  resourceId?: number;
  page?: number;
}

/** "Open PDF at page N" button shown under a chunk popover — renders nothing
 * if any id is missing (e.g. content indexed before this feature shipped). */
export default function OpenPdfButton({ appId, siloId, resourceId, page }: Props) {
  if (appId == null || siloId == null || resourceId == null || page == null) return null;
  return (
    <div className="px-3 py-2 border-t border-gray-100 dark:border-gray-600">
      <button
        onClick={() => openSourcePdf(appId, siloId, resourceId, page)}
        className="inline-flex items-center gap-1.5 text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-200 font-medium"
      >
        <FileText className="w-3.5 h-3.5" />
        Open PDF at page {page}
      </button>
    </div>
  );
}
