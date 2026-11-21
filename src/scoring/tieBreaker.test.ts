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
    // Force Initiative and Practicality to same score by making them each beat everyone else once differently
    // Simpler: start empty and craft two traits with equal wins
    const r = emptyResults();
    // Initiative beats everyone except Practicality
    for (const t of TRAITS) {
      if (t !== 'Initiative' && t !== 'Practicality') setWin(r, 'Initiative', t);
    }
    // Practicality beats everyone except Initiative
    for (const t of TRAITS) {
      if (t !== 'Initiative' && t !== 'Practicality') setWin(r, 'Practicality', t);
    }
    // Head-to-head: Practicality beats Initiative
    setWin(r, 'Practicality', 'Initiative');
    // Remaining traits: give Curiosity a unique ladder below
    const rest = TRAITS.filter((t) => t !== 'Initiative' && t !== 'Practicality');
    for (let i = 0; i < rest.length; i++) {
      for (let j = i + 1; j < rest.length; j++) {
        setWin(r, rest[i], rest[j]);
      }
    }

    const { ordered, ranks } = new TieBreaker(TRAITS, r).rankWithAverageRanks();
    expect(ordered[0]).toBe('Practicality');
    expect(ordered[1]).toBe('Initiative');
    expect(ranks.Practicality).toBe(1);
    expect(ranks.Initiative).toBe(2);
  });

  it('assigns average ranks for unresolved multi-way ties', () => {
    const r = emptyResults();
    // Three traits with identical scores and cyclic wins → unresolved → average ranks
    const trio = ['Initiative', 'Curiosity', 'Confidence'];
    // Each beats the same set of outsiders equally
    const outsiders = TRAITS.filter((t) => !trio.includes(t));
    for (const t of trio) {
      for (const o of outsiders) setWin(r, t, o);
    }
    // Cyclic: Initiative>Curiosity>Confidence>Initiative — equal mini-league wins (1 each)
    setWin(r, 'Initiative', 'Curiosity');
    setWin(r, 'Curiosity', 'Confidence');
    setWin(r, 'Confidence', 'Initiative');
    // Outsiders ladder among themselves
    for (let i = 0; i < outsiders.length; i++) {
      for (let j = i + 1; j < outsiders.length; j++) {
        setWin(r, outsiders[i], outsiders[j]);
      }
    }

    const { ranks } = new TieBreaker(TRAITS, r).rankWithAverageRanks();
    // Top 3 should share average rank 2 (positions 1,2,3)
    expect(ranks.Initiative).toBe(2);
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
    // Initiative: ranks 1 and 12 → avg 6.5
    expect(finalRanks.Initiative).toBeCloseTo(6.5, 5);
    // Distribution for Initiative: one Signature (1), one Situational (12)
    expect(distributionData.Initiative.Signature).toBe(0.5);
    expect(distributionData.Initiative.Situational).toBe(0.5);
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

describe('leadingStrengths', () => {
  it('returns just the #1 strength when there is no tie', async () => {
    const { leadingStrengths } = await import('./traits.js');
    expect(leadingStrengths(['Initiative', 'Empathy'], { Initiative: 1, Empathy: 2 })).toEqual(['Initiative']);
  });

  it('returns every strength sharing the top rank, so a tie for #1 shows all tied archetypes', async () => {
    const { leadingStrengths } = await import('./traits.js');
    const ordered = ['Curiosity', 'Initiative', 'Empathy', 'Fairness'];
    const ranks = { Curiosity: 1.5, Initiative: 1.5, Empathy: 3, Fairness: 4 };
    expect(leadingStrengths(ordered, ranks)).toEqual(['Curiosity', 'Initiative']);
  });
});

describe('legacy strength names in the question database', () => {
  it('scores the old "Courage" option value as Initiative, so existing answers still count for the renamed strength', async () => {
    const { buildSurveyResultsFromDb } = await import('./fromAnswers.js');
    const options = [
      { id: 'a', question_id: 1, construct: 'Courage' },
      { id: 'b', question_id: 1, construct: 'Empathy' },
    ];
    const results = buildSurveyResultsFromDb({ '1': 'a' }, [{ id: 1 }], options);
    expect(results.Initiative.Empathy).toBe(1);
  });

  it('is case-insensitive like the other construct names', async () => {
    const { buildSurveyResultsFromDb } = await import('./fromAnswers.js');
    const options = [
      { id: 'a', question_id: 1, construct: 'COURAGE' },
      { id: 'b', question_id: 1, construct: 'empathy' },
    ];
    expect(buildSurveyResultsFromDb({ '1': 'b' }, [{ id: 1 }], options).Empathy.Initiative).toBe(1);
  });
});
