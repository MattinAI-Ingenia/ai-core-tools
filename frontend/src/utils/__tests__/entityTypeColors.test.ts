import { describe, expect, it } from 'vitest';
import {
  colorForType,
  typeRingIcon,
  NEUTRAL_COLOR,
  PARTIAL_COLOR,
  TYPE_PALETTE,
} from '../entityTypeColors';

describe('colorForType', () => {
  it('is deterministic: the same type always gets the same color', () => {
    expect(colorForType('Machine')).toBe(colorForType('Machine'));
    expect(colorForType('Maquinaria@ES')).toBe(colorForType('Maquinaria@ES'));
  });

  it('maps every palette color to some type and stays inside the palette', () => {
    for (const type of ['Machine', 'Person', 'Proceso', 'Documento', 'Otro'] as const) {
      expect(TYPE_PALETTE).toContain(colorForType(type));
    }
  });

  it('spreads types across the palette instead of stacking on one slot', () => {
    const types = Array.from({ length: 12 }, (_, i) => `type-${i}`);
    const colors = new Set(types.map(colorForType));
    expect(colors.size).toBeGreaterThanOrEqual(4);
  });

  it('gives null/empty types the neutral partial gray', () => {
    expect(colorForType(null)).toBe(PARTIAL_COLOR);
    expect(colorForType(undefined)).toBe(PARTIAL_COLOR);
    expect(colorForType('')).toBe(PARTIAL_COLOR);
  });

  it('exports the neutral fill used when coloring is off', () => {
    expect(NEUTRAL_COLOR).toBe('#6366f1');
  });
});

describe('typeRingIcon', () => {
  it('builds an NVL overlayIcon with a stroked SVG data-URI of the color', () => {
    const ring = typeRingIcon('#22c55e');
    expect(ring.url.startsWith('data:image/svg+xml')).toBe(true);
    expect(decodeURIComponent(ring.url)).toContain('stroke="#22c55e"');
    expect(decodeURIComponent(ring.url)).not.toContain('fill="#22c55e"');
    expect(ring.position).toEqual([0, 0]);
    expect(ring.size).toBeGreaterThan(1);
  });

  it('differs between the cited-green and retrieved-indigo rings', () => {
    expect(typeRingIcon('#22c55e').url).not.toBe(typeRingIcon('#6366f1').url);
  });
});
