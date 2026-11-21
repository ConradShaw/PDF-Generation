import { describe, it, expect } from 'vitest';
import { TRAITS, categoryForRankNumber, colorForCategory, formatRank, emptyResults } from './traits.js';

describe('categoryForRankNumber', () => {
  it('puts ranks 1-3 in Signature, 4-6 Supporting, 7-9 Stretch and 10-12 Situational', () => {
    const cats = Array.from({ length: 12 }, (_, i) => categoryForRankNumber(i + 1));
    expect(cats).toEqual([...Array(3).fill('Signature'), ...Array(3).fill('Supporting'), ...Array(3).fill('Stretch'), ...Array(3).fill('Situational')]);
  });

  it('places a tied (fractional) rank by its average: 3.5 sits between Signature and Supporting, so it is Supporting', () => {
    expect(categoryForRankNumber(3)).toBe('Signature');
    expect(categoryForRankNumber(3.5)).toBe('Supporting');
    expect(categoryForRankNumber(6.5)).toBe('Stretch');
    expect(categoryForRankNumber(9.5)).toBe('Situational');
  });
});

describe('colorForCategory', () => {
  it('returns a valid CSS hex colour for every category, so no table cell is left unshaded', () => {
    for (const c of ['Signature', 'Supporting', 'Stretch', 'Situational'] as const) {
      expect(colorForCategory(c)).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  it('gives each of the four categories its own colour, so a table can be read at a glance', () => {
    const colours = (['Signature', 'Supporting', 'Stretch', 'Situational'] as const).map(colorForCategory);
    expect(new Set(colours).size).toBe(4);
  });
});

describe('formatRank', () => {
  it('shows whole ranks plainly and tied ranks with one decimal', () => {
    expect(formatRank(2)).toBe('2');
    expect(formatRank(1.5)).toBe('1.5');
    expect(formatRank(2.25)).toBe('2.3');
  });
});

describe('emptyResults', () => {
  it('has a zero for every ordered pair of different strengths and none for a strength against itself', () => {
    const r = emptyResults();
    expect(Object.keys(r).sort()).toEqual([...TRAITS].sort());
    for (const t of TRAITS) {
      expect(Object.keys(r[t])).toHaveLength(11);
      expect(r[t][t]).toBeUndefined();
      expect(Object.values(r[t]).every((v) => v === 0)).toBe(true);
    }
  });
});
