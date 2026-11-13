import { emptyResults, TRAITS } from './traits.js';
import { setWin } from './fromAnswers.js';
import { TieBreaker } from './tieBreaker.js';
import { calculateTeamRankings } from './teamRankings.js';
import { describe, expect, it } from 'vitest';

/** Build results where earlier traits beat all later ones (strict ladder). */
function ladderResults(order: string[]) {
  const results = emptyResults();
  for (let i = 0; i < order.length; i++) {
    for (let j = i + 1; j < order.length; j++) {
      setWin(results, order[i], order[j]);
    }
  }
  return results;
}

describe('TieBreaker', () => {
  it('ranks a strict ladder with no ties', () => {
    const order = [...TRAITS];
    const results = ladderResults(order);
    const { ordered, ranks } = new TieBreaker(TRAITS, results).rankWithAverageRanks();
    expect(ordered).toEqual(order);
    for (let i = 0; i < order.length; i++) {
      expect(ranks[order[i]]).toBe(i + 1);
    }
  });

  it('resolves a 2-way score tie via head-to-head', () => {
    const order = [...TRAITS];
    const results = ladderResults(order);
    // Force Courage and Practicality to same score by making them each beat everyone else once differently
    // Simpler: start empty and craft two traits with equal wins
    const r = emptyResults();
    // Courage beats everyone except Practicality
    for (const t of TRAITS) {
      if (t !== 'Courage' && t !== 'Practicality') setWin(r, 'Courage', t);
    }
    // Practicality beats everyone except Courage
    for (const t of TRAITS) {
      if (t !== 'Courage' && t !== 'Practicality') setWin(r, 'Practicality', t);
    }
    // Head-to-head: Practicality beats Courage
    setWin(r, 'Practicality', 'Courage');
    // Remaining traits: give Curiosity a unique ladder below
    const rest = TRAITS.filter((t) => t !== 'Courage' && t !== 'Practicality');
    for (let i = 0; i < rest.length; i++) {
      for (let j = i + 1; j < rest.length; j++) {
        setWin(r, rest[i], rest[j]);
      }
    }

    const { ordered, ranks } = new TieBreaker(TRAITS, r).rankWithAverageRanks();
    expect(ordered[0]).toBe('Practicality');
    expect(ordered[1]).toBe('Courage');
    expect(ranks.Practicality).toBe(1);
    expect(ranks.Courage).toBe(2);
  });

  it('assigns average ranks for unresolved multi-way ties', () => {
    const r = emptyResults();
    // Three traits with identical scores and cyclic wins → unresolved → average ranks
    const trio = ['Courage', 'Curiosity', 'Confidence'];
    // Each beats the same set of outsiders equally
    const outsiders = TRAITS.filter((t) => !trio.includes(t));
    for (const t of trio) {
      for (const o of outsiders) setWin(r, t, o);
    }
    // Cyclic: Courage>Curiosity>Confidence>Courage — equal mini-league wins (1 each)
    setWin(r, 'Courage', 'Curiosity');
    setWin(r, 'Curiosity', 'Confidence');
    setWin(r, 'Confidence', 'Courage');
    // Outsiders ladder among themselves
    for (let i = 0; i < outsiders.length; i++) {
      for (let j = i + 1; j < outsiders.length; j++) {
        setWin(r, outsiders[i], outsiders[j]);
      }
    }

    const { ranks } = new TieBreaker(TRAITS, r).rankWithAverageRanks();
    // Top 3 should share average rank 2 (positions 1,2,3)
    expect(ranks.Courage).toBe(2);
    expect(ranks.Curiosity).toBe(2);
    expect(ranks.Confidence).toBe(2);
  });
});

describe('calculateTeamRankings', () => {
  it('averages individual ranks and computes distribution', () => {
    const memberA = {
      ranks: Object.fromEntries(TRAITS.map((t, i) => [t, i + 1])),
    };
    const memberB = {
      ranks: Object.fromEntries(TRAITS.map((t, i) => [t, TRAITS.length - i])),
    };
    const { orderedTraits, finalRanks, distributionData } = calculateTeamRankings([
      memberA,
      memberB,
    ]);

    expect(orderedTraits).toHaveLength(12);
    expect(Object.keys(finalRanks)).toHaveLength(12);
    // Courage: ranks 1 and 12 → avg 6.5
    expect(finalRanks.Courage).toBeCloseTo(6.5, 5);
    // Distribution for Courage: one Signature (1), one Situational (12)
    expect(distributionData.Courage.Signature).toBe(0.5);
    expect(distributionData.Courage.Situational).toBe(0.5);
  });

  it('uses average-of-positions for tied team ranks', () => {
    const same = {
      ranks: Object.fromEntries(TRAITS.map((t, i) => [t, i + 1])),
    };
    const { finalRanks } = calculateTeamRankings([same, { ...same, ranks: { ...same.ranks } }]);
    // Identical members → same order as individual; integer ranks 1..12
    for (let i = 0; i < TRAITS.length; i++) {
      expect(finalRanks[TRAITS[i]]).toBe(i + 1);
    }
  });
});
