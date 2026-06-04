import { useState, useRef, useEffect } from 'react';
import Modal from './Modal';
import OneTimeLinkModal from './OneTimeLinkModal';
import { adminService } from '../../services/admin';
import type { LocalUserCreated } from '../../services/admin';
import { ApiError } from '../../services/api';
import { errorMessage } from '../../constants/messages';

export interface CreateLocalUserModalProps {
  readonly isOpen: boolean;
  readonly onClose: () => void;
  /** Called after a user is successfully created so the parent can refresh the list. */
  readonly onCreated: () => void;
}

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function CreateLocalUserModal({ isOpen, onClose, onCreated }: CreateLocalUserModalProps) {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  // Set when the server rejects the email as already-taken (409) so the email
  // field can be marked invalid even though it is well-formed.
  const [emailTaken, setEmailTaken] = useState(false);

  // Once a user is created we hold the result here and show the one-time link modal.
  const [createdUser, setCreatedUser] = useState<LocalUserCreated | null>(null);
  const [showLink, setShowLink] = useState(false);

  const emailInputRef = useRef<HTMLInputElement>(null);

  // Focus the email input when the modal opens.
  useEffect(() => {
    if (isOpen && !showLink) {
      // Defer one tick so the Modal has rendered.
      const id = setTimeout(() => emailInputRef.current?.focus(), 50);
      return () => clearTimeout(id);
    }
  }, [isOpen, showLink]);

  // Reset local state whenever the modal is closed from the outside.
  useEffect(() => {
    if (!isOpen) {
      setEmail('');
      setName('');
      setFormError(null);
      setIsSubmitting(false);
      setCreatedUser(null);
      setShowLink(false);
    }
  }, [isOpen]);

  function validate(): string | null {
    if (!email.trim()) return 'Email is required.';
    if (!EMAIL_PATTERN.test(email.trim())) return 'Please enter a valid email address.';
    if (!name.trim()) return 'Name is required.';
    return null;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);

    const ve = validate();
    if (ve) {
      setFormError(ve);
      return;
    }

    setIsSubmitting(true);
    setEmailTaken(false);
    try {
      const result = await adminService.createLocalUser(email.trim(), name.trim());
      setCreatedUser(result);
      setShowLink(true);
    } catch (err: unknown) {
      // Reliable status-based branch for the duplicate-email 409 case.
      if (err instanceof ApiError && err.status === 409) {
        setEmailTaken(true);
        setFormError('A user with this email address already exists.');
      } else {
        setFormError(errorMessage(err, 'Failed to create user.'));
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleLinkClose() {
    setShowLink(false);
    // Refresh the list only once the admin has acknowledged the one-time link.
    onCreated();
    onClose();
  }

  // When the link modal is active we render it on top instead of the form.
  if (showLink && createdUser) {
    return (
      <OneTimeLinkModal
        isOpen={true}
        onClose={handleLinkClose}
        token={createdUser.set_password_token}
        expiresAt={createdUser.expires_at}
        title={`User created — ${createdUser.email}`}
      />
    );
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create user" size="medium">
      <form onSubmit={handleSubmit} noValidate className="space-y-5">
        <p className="text-xs text-gray-500 dark:text-slate-400">Fields marked * are required.</p>
        {/* Inline error — always in DOM for screen readers (role=alert implies aria-live) */}
        <div
          id="create-user-error"
          role="alert"
          className={formError
            ? 'flex items-start gap-3 rounded-lg px-4 py-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
            : 'sr-only'}
        >
          {formError && (
            <p className="text-sm text-red-700 dark:text-red-300">{formError}</p>
          )}
        </div>

        <div>
          <label htmlFor="create-user-email" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1.5">
            Email <span aria-hidden="true">*</span>
          </label>
          <input
            ref={emailInputRef}
            id="create-user-email"
            type="email"
            value={email}
            onChange={(e) => { setEmail(e.target.value); setEmailTaken(false); }}
            required
            aria-required="true"
            aria-invalid={emailTaken || (formError !== null && (!email.trim() || !EMAIL_PATTERN.test(email.trim())))}
            aria-describedby="create-user-error"
            autoComplete="email"
            disabled={isSubmitting}
            placeholder="user@example.com"
            className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-lg text-sm text-gray-900 dark:text-slate-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        <div>
          <label htmlFor="create-user-name" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1.5">
            Full name <span aria-hidden="true">*</span>
          </label>
          <input
            id="create-user-name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            aria-required="true"
            aria-invalid={formError !== null && !name.trim()}
            aria-describedby="create-user-error"
            autoComplete="name"
            disabled={isSubmitting}
            placeholder="Jane Doe"
            className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-lg text-sm text-gray-900 dark:text-slate-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        <div className="flex justify-end gap-3 border-t border-gray-200 dark:border-slate-700 pt-4">
          <button
            type="button"
            onClick={onClose}
            disabled={isSubmitting}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSubmitting && (
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            )}
            {isSubmitting ? 'Creating…' : 'Create user'}
          </button>
        </div>
      </form>
    </Modal>
  );
}

export default CreateLocalUserModal;
