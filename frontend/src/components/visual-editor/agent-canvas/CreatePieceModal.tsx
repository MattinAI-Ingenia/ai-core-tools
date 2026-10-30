import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { apiService, type Agent, type DataStructure } from '../../../services/api';
import { useApiMutation } from '../../../hooks/useApiMutation';
import { MESSAGES, errorMessage } from '../../../constants/messages';
import type { SandboxServiceFormData, ServiceFormData } from '../../../types/services';
import Modal from '../../ui/Modal';
import SiloForm from '../../forms/SiloForm';
import SkillForm from '../../forms/SkillForm';
import MCPConfigForm from '../../forms/MCPConfigForm';
import DataStructureForm from '../../forms/DataStructureForm';
import MiddlewareForm from '../../forms/MiddlewareForm';
import AgentQuickCreateForm from '../../forms/AgentQuickCreateForm';
import ServiceWizard from '../../services/wizard/ServiceWizard';
import type { PieceKind } from './agentPorts';

export interface CreatePieceModalProps {
  readonly appId: number;
  /** The palette category the «+» was pressed on; decides which existing creation form opens. */
  readonly kind: PieceKind;
  readonly onClose: () => void;
  /** Called once the piece is saved. The caller closes the modal and connects the piece. */
  readonly onCreated: (kind: PieceKind, id: number, name: string) => void;
}

interface CreatedPiece {
  readonly id: number;
  readonly name: string;
}

/**
 * Creation modal opened from a palette «+». Reuses the forms and create calls
 * of each piece's own settings page: the service wizard for AI, embedding and
 * sandbox services, and the existing forms for the rest.
 */
export function CreatePieceModal({ appId, kind, onClose, onCreated }: Readonly<CreatePieceModalProps>) {
  const mutate = useApiMutation();
  const [blankParser, setBlankParser] = useState<DataStructure | null>(null);

  // Output parsers need the blank structure (with the parser options) before the form can open.
  useEffect(() => {
    if (kind !== 'parser') return;
    let cancelled = false;
    apiService
      .getOutputParser(appId, 0)
      .then((blank) => {
        if (!cancelled) setBlankParser(blank);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        toast.error(errorMessage(err, 'Failed to load parser options'));
        onClose();
      });
    return () => {
      cancelled = true;
    };
  }, [appId, kind, onClose]);

  /** Saves through the same toast-wrapped mutation the settings pages use, then reports the new piece. */
  async function save<T>(label: string, create: () => Promise<T>, pick: (created: T) => CreatedPiece): Promise<void> {
    const created = await mutate(create, {
      loading: MESSAGES.CREATING(label),
      success: MESSAGES.CREATED(label),
      error: (err) => errorMessage(err, MESSAGES.SAVE_FAILED(label)),
    });
    if (created === undefined) return;
    const piece = pick(created);
    onCreated(kind, piece.id, piece.name);
  }

  async function saveService(data: ServiceFormData | SandboxServiceFormData): Promise<void> {
    if (kind === 'ai_service') {
      await save(
        'AI service',
        () => apiService.createAIService(appId, data),
        (service) => ({ id: service.service_id, name: service.name }),
      );
    } else if (kind === 'embedding') {
      await save(
        'embedding service',
        () => apiService.createEmbeddingService(appId, data),
        (service) => ({ id: service.service_id, name: service.name }),
      );
    } else {
      await save(
        'sandbox service',
        () => apiService.createSandboxService(appId, data),
        (service) => ({ id: service.service_id, name: service.name }),
      );
    }
  }

  switch (kind) {
    case 'ai_service':
    case 'embedding':
    case 'sandbox':
      return (
        <ServiceWizard
          isOpen
          kind={kind === 'ai_service' ? 'ai' : kind}
          scope="app"
          appId={appId}
          onClose={onClose}
          onSave={saveService}
        />
      );

    case 'silo':
      return (
        <Modal isOpen onClose={onClose} title="Create New Silo" size="xlarge">
          <SiloForm
            onSubmit={(data) =>
              save(
                'silo',
                () => apiService.createSilo(appId, data),
                (silo) => ({ id: silo.silo_id, name: silo.name }),
              )
            }
            onCancel={onClose}
          />
        </Modal>
      );

    case 'skill':
      return (
        <Modal isOpen onClose={onClose} title="Create New Skill" size="large">
          <SkillForm
            skill={null}
            onSubmit={(data) =>
              save(
                'skill',
                () => apiService.createSkill(appId, data),
                (skill) => ({ id: skill.skill_id, name: skill.name }),
              )
            }
            onCancel={onClose}
          />
        </Modal>
      );

    case 'mcp':
      return (
        <Modal isOpen onClose={onClose} title="Create New MCP Config">
          <MCPConfigForm
            mcpConfig={null}
            onSubmit={(data) =>
              save(
                'MCP config',
                () => apiService.createMCPConfig(appId, data),
                (config) => ({ id: config.config_id, name: config.name }),
              )
            }
            onCancel={onClose}
          />
        </Modal>
      );

    case 'parser':
      return (
        <Modal isOpen onClose={onClose} title="Create New Data Structure" size="xlarge">
          {blankParser ? (
            <DataStructureForm
              dataStructure={{
                ...blankParser,
                fields: blankParser.fields ?? [],
                available_parsers: blankParser.available_parsers ?? [],
              }}
              onSubmit={(data) =>
                save(
                  'data structure',
                  () => apiService.createOutputParser(appId, data),
                  (parser) => ({ id: parser.parser_id, name: parser.name }),
                )
              }
              onCancel={onClose}
            />
          ) : (
            <p role="status" className="text-sm text-gray-500 dark:text-gray-400">
              {MESSAGES.LOADING('data structure')}
            </p>
          )}
        </Modal>
      );

    case 'middleware':
      return (
        <Modal isOpen onClose={onClose} title="Create New Middleware" size="large">
          <MiddlewareForm
            middleware={null}
            appId={appId}
            onSubmit={(data) =>
              save(
                'middleware',
                () => apiService.createMiddleware(appId, data),
                (middleware) => ({ id: middleware.middleware_id, name: middleware.name }),
              )
            }
            onCancel={onClose}
          />
        </Modal>
      );

    case 'agent':
      return (
        <Modal isOpen onClose={onClose} title="Create New Agent" size="medium">
          <AgentQuickCreateForm
            appId={appId}
            mode="modal"
            onCreated={(agent: Agent) => onCreated(kind, agent.agent_id, agent.name)}
            onCancel={onClose}
          />
        </Modal>
      );
  }
}
