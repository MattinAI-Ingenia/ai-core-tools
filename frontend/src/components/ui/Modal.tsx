import { type ReactNode, useEffect, useRef, useId } from 'react';
import { X } from 'lucide-react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  size?: 'small' | 'medium' | 'large' | 'xlarge';
}

const FOCUSABLE_SELECTORS = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

function Modal({ isOpen, onClose, title, children, size = 'large' }: Readonly<ModalProps>) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  // Capture the element that had focus when the modal opened so we can restore it.
  const returnFocusRef = useRef<Element | null>(null);

  // Define size classes
  const sizeClasses = {
    small: 'max-w-md max-h-96',
    medium: 'max-w-2xl max-h-[70vh]',
    large: 'max-w-4xl max-h-[80vh]',
    xlarge: 'max-w-6xl max-h-[90vh]',
  };

  useEffect(() => {
    if (!isOpen) return;

    // Capture current focus target so we can restore it on close.
    returnFocusRef.current = document.activeElement;

    // Move focus into the modal panel (first focusable element or the panel itself).
    const panel = panelRef.current;
    if (panel) {
      const first = panel.querySelector<HTMLElement>(FOCUSABLE_SELECTORS);
      (first ?? panel).focus();
    }

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }

      if (e.key !== 'Tab') return;

      const currentPanel = panelRef.current;
      if (!currentPanel) return;

      const focusables = Array.from(
        currentPanel.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS),
      ).filter((el) => !el.closest('[hidden]'));

      if (focusables.length === 0) {
        e.preventDefault();
        return;
      }

      const first = focusables[0];
      const last = focusables[focusables.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      // Restore focus to the element that was active before the modal opened.
      if (returnFocusRef.current instanceof HTMLElement) {
        returnFocusRef.current.focus();
      }
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        aria-hidden="true"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
          className={`relative bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full overflow-hidden focus:outline-none ${sizeClasses[size]}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-slate-600 bg-gray-50 dark:bg-slate-700 sticky top-0 z-10">
            <h3 id={titleId} className="text-lg font-semibold text-gray-900 dark:text-slate-100">
              {title}
            </h3>
            <button
              onClick={onClose}
              aria-label="Close"
              className="text-gray-400 hover:text-gray-600 dark:text-slate-400 dark:hover:text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
            >
              <X className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(80vh - 88px)' }}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Modal;
