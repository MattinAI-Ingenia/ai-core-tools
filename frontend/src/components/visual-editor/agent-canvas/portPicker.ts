/**
 * Pure filtering for a port's «+» picker: which existing pieces the port
 * accepts, which are excluded because they are already connected to that very
 * port, and which are disabled (retired services stay listed but cannot be
 * picked).
 */

import type { PieceKind, PortDef, PortId } from './agentPorts';
import type { CanvasPiece } from './useAgentCanvasData';

/** One entry of the port picker: an acceptable piece plus whether it can be picked. */
export interface PortPickerEntry {
  readonly piece: CanvasPiece;
  /** Retired services cannot connect; the picker shows them disabled. */
  readonly disabled: boolean;
}

/**
 * Pieces a port's picker offers: the kinds it accepts, excluding the ones
 * already connected to this port (the same piece on another port stays), and
 * for the Video port only AI services with video support.
 */
export function piecesForPort(
  port: PortDef,
  pieces: readonly CanvasPiece[],
  connected: ReadonlyArray<{ port: PortId; kind: PieceKind; id: number }>,
): PortPickerEntry[] {
  const onPort = new Set(
    connected.filter((entry) => entry.port === port.id).map((entry) => `${entry.kind}:${entry.id}`),
  );
  return pieces
    .filter((piece) => port.accepts.includes(piece.kind))
    .filter((piece) => !(port.id === 'video' && piece.kind === 'ai_service' && piece.supportsVideo !== true))
    .filter((piece) => !onPort.has(`${piece.kind}:${piece.id}`))
    .map((piece) => ({ piece, disabled: piece.lifecycleStatus === 'retired' }));
}
