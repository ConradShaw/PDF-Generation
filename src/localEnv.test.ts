import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadLocalEnv } from './localEnv.js';

const dir = mkdtempSync(join(tmpdir(), 'localenv-'));
const file = join(dir, '.env');
const NAMES = ['LOCALENV_TEST_A', 'LOCALENV_TEST_B'];

afterEach(() => {
  for (const n of NAMES) delete process.env[n];
});

describe('loadLocalEnv', () => {
  it('loads the settings from a .env file so npm run dev works without extra flags', () => {
    writeFileSync(file, 'LOCALENV_TEST_A=from-file\nLOCALENV_TEST_B="quoted value"\n');
    expect(loadLocalEnv(file)).toBe(true);
    expect(process.env.LOCALENV_TEST_A).toBe('from-file');
    expect(process.env.LOCALENV_TEST_B).toBe('quoted value');
  });

  it('never overrides a variable that is already set for real: a host\'s environment beats a leftover file', () => {
    process.env.LOCALENV_TEST_A = 'from-the-host';
    writeFileSync(file, 'LOCALENV_TEST_A=from-file\n');
    loadLocalEnv(file);
    expect(process.env.LOCALENV_TEST_A).toBe('from-the-host');
  });

  it('does nothing, and does not throw, when there is no .env file, as in production', () => {
    expect(loadLocalEnv(join(dir, 'missing.env'))).toBe(false);
    expect(process.env.LOCALENV_TEST_A).toBeUndefined();
  });
});

process.on('exit', () => rmSync(dir, { recursive: true, force: true }));
