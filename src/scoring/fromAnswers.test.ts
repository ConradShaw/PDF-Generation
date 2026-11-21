import { describe, it, expect } from 'vitest';
import { buildSurveyResultsFromDb, setWin } from './fromAnswers.js';
import { emptyResults } from './traits.js';

/** One question between two strengths; option ids are `${q}a` and `${q}b`. */
const q = (id: number, a: string, b: string) => ({
  question: { id },
  options: [
    { id: `${id}a`, question_id: id, construct: a },
    { id: `${id}b`, question_id: id, construct: b },
  ],
});
const run = (answers: Record<string, string>, ...qs: ReturnType<typeof q>[]) =>
  buildSurveyResultsFromDb(answers, qs.map((x) => x.question), qs.flatMap((x) => x.options));
const noWins = (r: ReturnType<typeof emptyResults>) => Object.values(r).every((row) => Object.values(row).every((v) => v === 0));

describe('buildSurveyResultsFromDb: turning answers into head-to-head wins', () => {
  it('credits the chosen strength with a win over the one it was paired with', () => {
    const r = run({ '1': '1a' }, q(1, 'Curiosity', 'Empathy'));
    expect(r.Curiosity.Empathy).toBe(1);
    expect(r.Empathy.Curiosity).toBe(0);
  });

  it('credits the second option when that is the one chosen', () => {
    const r = run({ '1': '1b' }, q(1, 'Curiosity', 'Empathy'));
    expect(r.Empathy.Curiosity).toBe(1);
    expect(r.Curiosity.Empathy).toBe(0);
  });

  it('ignores an unanswered question rather than guessing', () => {
    expect(noWins(run({}, q(1, 'Curiosity', 'Empathy')))).toBe(true);
  });

  it('ignores an answer whose option id belongs to neither option', () => {
    expect(noWins(run({ '1': 'nonsense' }, q(1, 'Curiosity', 'Empathy')))).toBe(true);
  });

  it('ignores a question that does not have exactly two options', () => {
    const three = { question: { id: 1 }, options: [
      { id: '1a', question_id: 1, construct: 'Curiosity' },
      { id: '1b', question_id: 1, construct: 'Empathy' },
      { id: '1c', question_id: 1, construct: 'Fairness' },
    ] };
    expect(noWins(buildSurveyResultsFromDb({ '1': '1a' }, [three.question], three.options))).toBe(true);
  });

  it('ignores a pairing that names something that is not one of the 12 strengths', () => {
    expect(noWins(run({ '1': '1a' }, q(1, 'Curiosity', 'Bravery')))).toBe(true);
  });

  it('ignores a question that pairs a strength with itself', () => {
    expect(noWins(run({ '1': '1a' }, q(1, 'Curiosity', 'curiosity')))).toBe(true);
  });

  it('reads strength names case-insensitively, as the question data is not consistently capitalised', () => {
    expect(run({ '1': '1a' }, q(1, ' curiosity ', 'EMPATHY')).Curiosity.Empathy).toBe(1);
  });

  it('accumulates wins across questions', () => {
    const r = run({ '1': '1a', '2': '2a' }, q(1, 'Curiosity', 'Empathy'), q(2, 'Curiosity', 'Fairness'));
    expect(r.Curiosity.Empathy).toBe(1);
    expect(r.Curiosity.Fairness).toBe(1);
  });
});

describe('setWin', () => {
  it('records the win and clears the opposite result, so a pair can never be won by both', () => {
    const r = emptyResults();
    r.Empathy.Curiosity = 1;
    setWin(r, 'Curiosity', 'Empathy');
    expect(r.Curiosity.Empathy).toBe(1);
    expect(r.Empathy.Curiosity).toBe(0);
  });

  it('does nothing for a strength paired with itself or an unknown strength', () => {
    const r = emptyResults();
    setWin(r, 'Curiosity', 'Curiosity');
    setWin(r, 'Nobody', 'Curiosity');
    expect(noWins(r)).toBe(true);
  });
});
