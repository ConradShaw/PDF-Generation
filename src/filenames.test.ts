import { describe, it, expect } from 'vitest';
import { fileSafe } from './filenames.js';
import { profileFilename } from './templates/profile.js';
import { careerFilename } from './templates/career.js';

describe('fileSafe', () => {
  it('folds accents so names like José and Zoë keep working in storage keys', () => {
    expect(fileSafe('José')).toBe('Jose');
    expect(fileSafe('Zoë Müller')).toBe('Zoe_Muller');
  });

  it('removes path separators and other unsafe characters, so a name can never escape its folder', () => {
    expect(fileSafe('../../etc/passwd')).not.toContain('/');
    expect(fileSafe('a b\\c:d')).toBe('a_b_c_d');
  });

  it('collapses a run of unsafe characters into a single underscore', () => {
    expect(fileSafe('a !!!   b')).toBe('a_b');
  });

  it('does not start or end with underscores left over from trimmed characters', () => {
    expect(fileSafe('!!Ada!!')).toBe('Ada');
    expect(fileSafe('__x__')).toBe('x');
  });

  it('falls back to a placeholder when nothing ASCII remains', () => {
    expect(fileSafe('李明')).toBe('Report');
    expect(fileSafe('   ')).toBe('Report');
  });
});

describe('report filenames', () => {
  it('are ASCII-only for any customer name', () => {
    for (const make of [profileFilename, careerFilename]) {
      const name = make('Zoë', 'Ñandú/Ω', '2026-11-19');
      expect(name).toMatch(/^[A-Za-z0-9._-]+$/);
    }
  });
});
