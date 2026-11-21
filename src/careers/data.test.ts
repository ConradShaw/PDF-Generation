import { describe, it, expect } from 'vitest';
import { TRAITS } from '../scoring/traits.js';
import {
  archetypeFor,
  assertOwnerDataReady,
  occupations,
  ownerDataProblems,
} from './data.js';

describe('ownerDataProblems / assertOwnerDataReady', () => {
  const ready = { archetypes: { placeholder: false }, mapping: { placeholder: false } };

  it('reports nothing when the owner has supplied both inputs', () => {
    expect(ownerDataProblems(ready)).toEqual([]);
  });

  it('flags placeholder archetype text so customers never receive "[PLACEHOLDER]" copy', () => {
    expect(ownerDataProblems({ ...ready, archetypes: { placeholder: true } })).toHaveLength(1);
  });

  it('flags an unconfirmed strength mapping', () => {
    expect(ownerDataProblems({ ...ready, mapping: { placeholder: true } })).toHaveLength(1);
  });

  it('refuses in production when inputs are placeholders, but allows development to run end to end', () => {
    const placeholder = { archetypes: { placeholder: true }, mapping: { placeholder: false } };
    expect(() => assertOwnerDataReady(true, placeholder)).toThrow(/placeholder/i);
    expect(() => assertOwnerDataReady(false, placeholder)).not.toThrow();
  });
});

describe('archetypeFor / owner data messages', () => {
  it('refuses to invent text for a strength that has none', () => {
    expect(() => archetypeFor('Bravery')).toThrow(/Bravery/);
  });

  it('names the file that needs fixing, so whoever sees the error knows where to look', () => {
    const msgs = ownerDataProblems({ archetypes: { placeholder: true }, mapping: { placeholder: true } });
    expect(msgs.join(' ')).toMatch(/archetypes\.json/);
    expect(msgs.join(' ')).toMatch(/ssm-mapping\.json/);
    expect(() => assertOwnerDataReady(true, { archetypes: { placeholder: true }, mapping: { placeholder: false } })).toThrow(/archetypes\.json/);
  });
});

describe('bundled O*NET data', () => {
  it('covers roughly a thousand occupations, each with all 12 strength requirements in 0-100', () => {
    expect(occupations.length).toBeGreaterThan(800);
    for (const occ of occupations) {
      for (const t of TRAITS) {
        expect(occ.requirements[t]).toBeGreaterThanOrEqual(0);
        expect(occ.requirements[t]).toBeLessThanOrEqual(100);
      }
    }
  });

  it('has a Work Archetype entry for every strength', () => {
    for (const t of TRAITS) expect(archetypeFor(t).length).toBeGreaterThan(0);
  });
});
