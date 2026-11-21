import { describe, it, expect } from 'vitest';
import { TRAITS } from '../src/scoring/traits.js';
import { renderProfileHtml } from '../src/templates/profile.js';
import { renderCareerHtml, summarySentence } from '../src/templates/career.js';

const ordered = [...TRAITS];
const ranks = Object.fromEntries(TRAITS.map((t, i) => [t, i + 1]));

describe('free Strength Profile', () => {
  const html = renderProfileHtml({
    first: 'Ada',
    last: 'Lovelace',
    dateStr: '2026-11-19',
    orderedTraits: ordered,
    ranks,
    archetypes: [{ trait: 'Initiative', text: 'THE-INITIATIVE-ARCHETYPE' }],
    unlockUrl: 'https://shawsight.test/career-match/unlock?assessment=abc',
  });

  it('shows the Work Archetype text for the #1 strength only: the rest is what the paid upgrade is for', () => {
    expect(html).toContain('THE-INITIATIVE-ARCHETYPE');
    expect(html.match(/THE-INITIATIVE-ARCHETYPE/g)).toHaveLength(1);
  });

  it('shows every tied archetype when strengths tie for #1, naming each one', () => {
    const tied = renderProfileHtml({
      first: 'A',
      last: 'B',
      dateStr: '2026-11-19',
      orderedTraits: ordered,
      ranks,
      archetypes: [
        { trait: 'Initiative', text: 'TEXT-INITIATIVE' },
        { trait: 'Practicality', text: 'TEXT-PRACTICALITY' },
      ],
      unlockUrl: 'https://x.test/',
    });
    expect(tied).toContain('TEXT-INITIATIVE');
    expect(tied).toContain('TEXT-PRACTICALITY');
    expect(tied).toContain('tied');
    expect(html).not.toContain('tied'); // a single #1 is not described as a tie
  });

  it('lists all 12 ranked strengths', () => {
    for (const t of TRAITS) expect(html).toContain(t);
  });

  it('carries the unlock link that sends the customer to pay without retaking the assessment', () => {
    expect(html).toContain('https://shawsight.test/career-match/unlock?assessment=abc');
  });

  it('does not leak any career match content into the free report', () => {
    expect(html).not.toContain('Chief Executives');
  });

  it('escapes owner-supplied archetype text', () => {
    const unsafe = renderProfileHtml({
      first: 'A',
      last: 'B',
      dateStr: '2026-11-19',
      orderedTraits: ordered,
      ranks,
      archetypes: [{ trait: 'Initiative', text: '<script>x</script>' }],
      unlockUrl: 'https://x.test/',
    });
    expect(unsafe).not.toContain('<script>x</script>');
  });
});

describe('paid Career Matches report', () => {
  const matches = [
    { code: '11-1011.00', title: 'Chief Executives', rank: 1, stars: 5 },
    { code: '11-2021.00', title: 'Marketing Managers', rank: 2, stars: 5 },
    { code: '13-1111.00', title: 'Management Analysts', rank: 3, stars: 4 },
  ];
  const html = renderCareerHtml({
    first: 'Ada',
    last: 'Lovelace',
    dateStr: '2026-11-19',
    orderedTraits: ordered,
    ranks,
    riasec: { R: 5, I: 25, A: 10, S: 10, E: 15, C: 5 },
    matches,
    archetypes: TRAITS.map((t) => ({ trait: t, text: `ARCHETYPE-OF-${t}` })),
    attribution: 'O*NET 31.0 Database by USDOL/ETA. CC BY 4.0.',
  });

  it('lists every matched occupation', () => {
    for (const m of matches) expect(html).toContain(m.title);
  });

  it('shows stars, never percentages: stars are the customer-facing rating', () => {
    const visibleText = html.replace(/<style>[\s\S]*?<\/style>/g, '').replace(/<[^>]*>/g, ' ');
    expect(visibleText).not.toMatch(/\d\s?%/);
    expect(html).toContain('data-stars="5"');
  });

  it('lists occupations in rank order, with the rank number, even if given out of order', () => {
    const shuffled = renderCareerHtml({
      first: 'A', last: 'B', dateStr: '2026-11-19', orderedTraits: ordered, ranks,
      riasec: { R: 5, I: 25, A: 10, S: 10, E: 15, C: 5 },
      matches: [...matches].reverse(),
      archetypes: TRAITS.map((t) => ({ trait: t, text: t })),
      attribution: 'x',
    });
    expect(shuffled.indexOf('Chief Executives')).toBeLessThan(shuffled.indexOf('Marketing Managers'));
    expect(shuffled.indexOf('Marketing Managers')).toBeLessThan(shuffled.indexOf('Management Analysts'));
    expect(shuffled).toContain('<td class="rank">1</td>');
  });

  it('opens the matches with a one-sentence summary of the top two interest areas and the top three strengths', () => {
    expect(summarySentence({ R: 5, I: 25, A: 10, S: 10, E: 15, C: 5 }, ['Confidence', 'Empathy', 'Foresight'])).toBe(
      'Your strongest career interests are in Investigative and Enterprising activities, while your characteristic strengths suggest particular alignment with roles requiring Confidence, Empathy and Foresight.',
    );
    expect(html).toContain('Your strongest career interests are in Investigative and Enterprising activities');
  });

  it('drops the interest clause for a user with no interest preference, rather than naming arbitrary areas', () => {
    expect(summarySentence({ R: 15, I: 15, A: 15, S: 15, E: 15, C: 15 }, ['Confidence', 'Empathy', 'Foresight'])).toBe(
      'Your characteristic strengths suggest particular alignment with roles requiring Confidence, Empathy and Foresight.',
    );
  });

  it('states that the list measures fit, not competence or readiness', () => {
    expect(html).toContain('assesses occupational fit, not competence');
    expect(html).toContain('determine your readiness');
  });

  it('prints the O*NET attribution the license requires, including that changes were made', () => {
    expect(html).toContain('CC BY 4.0');
    expect(html).toContain('Interest Profiler');
    expect(html).toContain('Changes were made');
  });

  it('shows the Work Archetype of all 12 strengths, in rank order', () => {
    for (const t of TRAITS) expect(html).toContain(`ARCHETYPE-OF-${t}`);
    expect(html.indexOf('ARCHETYPE-OF-Initiative')).toBeLessThan(html.indexOf('ARCHETYPE-OF-Empathy'));
  });

  it('contains everything in the free profile (archetype for #1, strengths with descriptions), so paying never means getting less', () => {
    expect(html).toContain('ARCHETYPE-OF-Initiative');
    for (const t of TRAITS) expect(html).toContain(t);
    expect(html).toContain('The capacity to'); // strength descriptions
  });
});

describe('career report: details a customer reads', () => {
  const render = (over: Partial<Parameters<typeof renderCareerHtml>[0]> = {}) =>
    renderCareerHtml({
      first: 'Ada', last: 'Lovelace', dateStr: '2026-11-19', orderedTraits: ordered, ranks,
      riasec: { R: 0, I: 20, A: 10, S: 5, E: 20, C: 0 },
      matches: [
        { code: 'a', title: 'Five', rank: 1, stars: 5 },
        { code: 'b', title: 'Three', rank: 2, stars: 3 },
        { code: 'c', title: 'One', rank: 3, stars: 1 },
      ],
      archetypes: TRAITS.map((t) => ({ trait: t, text: `ARCHETYPE-OF-${t}` })),
      attribution: 'x',
      ...over,
    });

  it('fills exactly as many stars as the rating, and greys the rest, for every row', () => {
    const html = render();
    for (const stars of [5, 3, 1]) {
      const row = html.match(new RegExp(`<span data-stars="${stars}">(.*?)</span>`))![1];
      expect(row.match(/#ba9354/g) ?? []).toHaveLength(stars);
      expect(row.match(/#d9d9d9/g) ?? []).toHaveLength(5 - stars);
    }
  });

  it('draws each interest bar in proportion to the strongest area: the top area is full, half-strength is half, none is empty', () => {
    const html = render();
    const width = (area: string) => Number(html.match(new RegExp(`${area}</div>\\s*<div class="bar-track"><div class="bar-fill" style="width:(\\d+)%`))![1]);
    expect(width('Investigative')).toBe(100);
    expect(width('Enterprising')).toBe(100);
    expect(width('Artistic')).toBe(50);
    expect(width('Social')).toBe(25);
    expect(width('Realistic')).toBe(0);
  });

  it('draws empty bars, not NaN, for a customer whose interest scores are all zero', () => {
    const html = render({ riasec: { R: 0, I: 0, A: 0, S: 0, E: 0, C: 0 } });
    expect(html).not.toMatch(/width:NaN/);
    expect(html.match(/bar-fill" style="width:0%/g)).toHaveLength(6);
  });

  it('summarises with exactly the customer\'s top three strengths, not more', () => {
    const html = render();
    expect(html).toContain(`requiring ${ordered[0]}, ${ordered[1]} and ${ordered[2]}.`);
  });

  it('names the customer in full', () => {
    expect(render()).toContain('Prepared for: Ada Lovelace');
  });

  it('numbers the archetypes 1 to 12 in rank order, each with its strength name', () => {
    const html = render();
    TRAITS.forEach((t, i) => expect(html).toContain(`${i + 1}. ${t}:`));
  });

  it('describes every strength in the ranked table', () => {
    expect(render()).toContain('The capacity to understand and connect');
  });
});

describe('summarySentence', () => {
  const eq = { R: 0, I: 10, A: 0, S: 0, E: 10, C: 0 };

  it('lists one, two or three strengths in plain English: "A", "A and B", "A, B and C"', () => {
    expect(summarySentence(eq, ['Empathy'])).toContain('requiring Empathy.');
    expect(summarySentence(eq, ['Empathy', 'Foresight'])).toContain('requiring Empathy and Foresight.');
    expect(summarySentence(eq, ['Empathy', 'Foresight', 'Courage'])).toContain('requiring Empathy, Foresight and Courage.');
  });

  it('breaks a tie between interest areas in the fixed R-I-A-S-E-C order, so the sentence never changes between runs', () => {
    expect(summarySentence(eq, ['Empathy'])).toContain('Investigative and Enterprising activities');
  });

  it('keeps the fixed R-I-A-S-E-C order across a three-way tie, taking the first two', () => {
    expect(summarySentence({ R: 0, I: 10, A: 10, S: 0, E: 10, C: 0 }, ['Empathy'])).toContain('Investigative and Artistic activities');
  });

  it('puts the stronger area first', () => {
    expect(summarySentence({ R: 0, I: 5, A: 0, S: 0, E: 15, C: 9 }, ['Empathy'])).toContain('Enterprising and Conventional activities');
  });
});

describe('free profile: single and tied top strength', () => {
  const render = (archetypes: Array<{ trait: string; text: string }>) =>
    renderProfileHtml({ first: 'Ada', last: 'Lovelace', dateStr: '2026-11-19', orderedTraits: ordered, ranks, archetypes, unlockUrl: 'https://x.test/' });

  it('introduces a single top strength by name, without any "tied" wording and without labelling the paragraph', () => {
    const html = render([{ trait: 'Empathy', text: 'EMPATHY-TEXT' }]);
    expect(html).toContain('Your highest-ranked strength is <b>Empathy</b>');
    expect(html).not.toContain('tied');
    expect(html).not.toContain('<b>Empathy:</b>');
  });

  it('introduces tied strengths together and labels each paragraph with its strength', () => {
    const html = render([{ trait: 'Empathy', text: 'EMPATHY-TEXT' }, { trait: 'Fairness', text: 'FAIRNESS-TEXT' }]);
    expect(html).toContain('<b>Empathy</b> and <b>Fairness</b>');
    expect(html).toContain('<b>Empathy:</b> EMPATHY-TEXT');
    expect(html).toContain('<b>Fairness:</b> FAIRNESS-TEXT');
  });

  it('lists every benefit of unlocking, each as its own bullet', () => {
    const html = render([{ trait: 'Empathy', text: 't' }]);
    expect(html.match(/<li>/g)).toHaveLength(4);
  });

  it('names the customer in full and describes every strength', () => {
    const html = render([{ trait: 'Empathy', text: 't' }]);
    expect(html).toContain('Prepared for: Ada Lovelace');
    expect(html).toContain('The capacity to understand and connect');
  });
});
