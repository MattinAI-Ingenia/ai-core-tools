/**
 * Failing regression test reproducer.
 *
 * How to run (Unix/macOS):
 *   VITE_API_BASE_URL=http://localhost:8020 npx vitest frontend/src/core/__tests__/ConfigService.test.ts --run
 *
 * How to run (Windows PowerShell):
 *   $env:VITE_API_BASE_URL='http://localhost:8020'; npx vitest frontend/src/core/__tests__/ConfigService.test.ts --run
 *
 * This test asserts that the build-time env `import.meta.env.VITE_API_BASE_URL` is used when
 * `globalThis.__RUNTIME_CONFIG__.VITE_API_BASE_URL` is an empty string. On the current codebase
 * this WILL FAIL because the runtime empty string wins due to use of the `??` operator.
 */

import { afterEach, describe, expect, it } from 'vitest';
import { configService } from '../ConfigService';

describe('ConfigService.getApiBaseUrl - regression reproducer', () => {
  afterEach(() => {
    // Clean up runtime config after each test to avoid polluting other tests.
    delete (globalThis as any).__RUNTIME_CONFIG__;
  });

  it('should prefer build-time VITE_API_BASE_URL when runtime placeholder is empty string', () => {
    // Simulate Docker/runtime placeholder providing an empty string.
    (globalThis as any).__RUNTIME_CONFIG__ = { VITE_API_BASE_URL: '' };

    // Derive the expected value from the build-time env (Vitest picks this up from process.env)
    const expected = ((import.meta.env as any).VITE_API_BASE_URL as string) || 'http://localhost:8020';

    const actual = configService.getApiBaseUrl();

    // Expect build-time env to be used. Current codebase returns '' instead, so this assertion fails.
    expect(actual).toBe(expected);
  });
});
