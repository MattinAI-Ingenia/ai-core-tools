import { type ReactNode, useEffect, useRef, useId, useState } from 'react';
import { X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: ReactNode;
  size?: 'small' | 'medium' | 'large' | 'xlarge';
  /** Pinned below the scrollable content so action buttons stay reachable. */
  footer?: ReactNode;
}

const FOCUSABLE_SELECTORS = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

// Open modals, bottom to top. Only the last one handles Esc/Tab; the others are aria-hidden.
const openModalStack: string[] = [];
const stackListeners = new Set<() => void>();

function pushModal(id: string) {
  openModalStack.push(id);
  stackListeners.forEach((l) => l());
}

function removeModal(id: string) {
  const idx = openModalStack.indexOf(id);
  if (idx !== -1) openModalStack.splice(idx, 1);
  stackListeners.forEach((l) => l());
}

/** True while any Modal is open: other overlays (e.g. a side panel) must leave Esc and Tab to it. */
export function hasOpenModal(): boolean {
  return openModalStack.length > 0;
}

function isTopModal(id: string): boolean {
  return openModalStack[openModalStack.length - 1] === id;
}

function Modal({ isOpen, onClose, title, subtitle, children, size = 'large', footer }: Readonly<ModalProps>) {
  const { t } = useTranslation('common');
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  const returnFocusRef = useRef<Element | null>(null);
  // Stable ref so the effect depends on `isOpen` only — avoids re-running (and re-stealing focus)
  // when the parent passes a new onClose reference on each render (e.g. while user types).
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const [covered, setCovered] = useState(false);
  // Position in the open-modal stack. Paint order must follow interaction
  // order: every modal is `fixed` with no z-index of its own, so the one
  // later in the DOM wins visually. When DOM order disagrees with the stack
  // (a modal opened earlier but rendered later in the DOM), the covered
  // modal paints above the interactive one: the interaction target looks
  // gone and clicks close the wrong modal.
  const [depth, setDepth] = useState(0);

  const sizeClasses = {
    small: 'max-w-md max-h-96',
    medium: 'max-w-2xl max-h-[70vh]',
    large: 'max-w-4xl max-h-[80vh]',
    xlarge: 'max-w-6xl max-h-[90vh]',
  };
  const contentMaxHeights = {
    small: 'calc(24rem - 88px)',
    medium: 'calc(70vh - 88px)',
    large: 'calc(80vh - 88px)',
    xlarge: 'calc(90vh - 88px)',
  };

  useEffect(() => {
    if (!isOpen) return;

    returnFocusRef.current = document.activeElement;

    pushModal(titleId);
    const syncStackPosition = () => {
      setCovered(openModalStack.length > 0 && !isTopModal(titleId));
      setDepth(Math.max(openModalStack.indexOf(titleId), 0));
    };
    stackListeners.add(syncStackPosition);
    syncStackPosition();

    const panel = panelRef.current;
    if (panel) {
      const first = panel.querySelector<HTMLElement>(FOCUSABLE_SELECTORS);
      (first ?? panel).focus();
    }

    function handleKeyDown(e: KeyboardEvent) {
      if (!isTopModal(titleId)) return;
      if (e.key === 'Escape') {
        e.stopPropagation();
        onCloseRef.current();
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
      stackListeners.delete(syncStackPosition);
      removeModal(titleId);
      const target = returnFocusRef.current;
      if (target instanceof HTMLElement) {
        if (openModalStack.length > 0) {
          // A lower modal is still open but still aria-hidden until it re-renders; focus afterwards.
          requestAnimationFrame(() => target.focus());
        } else {
          target.focus();
        }
      }
    };
  }, [isOpen, titleId]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 overflow-y-auto"
      style={{ zIndex: 50 + depth }}
      // aria-hidden only, never `inert`: a covered modal's subtree often holds
      // the next modal (modals render nested inside their opener), and inert
      // would make that top modal invisible to hit-testing — its clicks pass
      // through to the page behind it. Interaction blocking is guaranteed by
      // the top modal's full-screen overlay at a higher z-index instead.
      aria-hidden={covered ? true : undefined}
    >
      <div
        className="fixed inset-0 bg-ink/40 dark:bg-black/60 transition-opacity"
        aria-hidden="true"
        onClick={onClose}
      />

      <div className="flex min-h-full items-center justify-center p-4">
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
          className={`relative bg-surface dark:bg-surface-dark border border-line dark:border-line-dark rounded-lg shadow-[0_14px_40px_rgba(32,28,20,0.14)] dark:shadow-[0_14px_40px_rgba(0,0,0,0.55)] w-full overflow-hidden focus:outline-none ${sizeClasses[size]}`}
        >
          <div className="flex items-center justify-between px-6 py-5 border-b border-line dark:border-line-dark bg-surface dark:bg-surface-dark sticky top-0 z-10">
            <div>
              <h3 id={titleId} className="font-display text-lg font-normal tracking-[-0.02em] text-fg dark:text-fg-dark">
                {title}
              </h3>
              {subtitle && (
                <p className="mt-1 text-sm text-fg-secondary dark:text-fg-secondary-dark">
                  {subtitle}
                </p>
              )}
            </div>
            <button
              onClick={onClose}
              aria-label={t('close')}
              className="p-1 text-fg-tertiary hover:text-fg hover:bg-surface-hover dark:text-fg-tertiary-dark dark:hover:text-fg-dark dark:hover:bg-surface-hover-dark transition-colors focus:outline-none focus-visible:outline-none rounded-lg"
            >
              <X className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>

          <div className="p-6 overflow-y-auto" style={{ maxHeight: contentMaxHeights[size] }}>
            {children}
          </div>

          {footer && (
            <div className="flex justify-end gap-3 p-4 border-t border-line dark:border-line-dark shrink-0">
              {footer}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Modal;
