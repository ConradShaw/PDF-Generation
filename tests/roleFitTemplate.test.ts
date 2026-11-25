import { describe, it, expect } from 'vitest';
import { TRAITS } from '../src/scoring/traits.js';
import { renderRoleFitHtml, roleFitFilename } from '../src/templates/roleFit.js';
import { targetProfile, rankCandidates } from '../src/rolefit/engine.js';
import type { Occupation } from '../src/careers/types.js';

const occupation: Occupation = {
  code: '11-1011.00', title: 'Chief Executives', profileDate: '12/2025',
  riasec: { R: 1, I: 1, A: 1, S: 1, E: 1, C: 1 },
  requirements: Object.fromEntries(TRAITS.map((t, i) => [t, 100 - i * 5])) as Occupation['requirements'],
};
const profile = targetProfile(occupation);
const ranksOf = (order: readonly string[]) => Object.fromEntries(order.map((t, i) => [t, i + 1]));
const candidates = rankCandidates(
  [
    { id: '1', name: 'Alice Fit', ranks: ranksOf(TRAITS) },
    { id: '2', name: 'Bob Far', ranks: ranksOf([...TRAITS].reverse()) },
  ],
  profile,
);
const render = (over: Partial<Parameters<typeof renderRoleFitHtml>[0]> = {}) =>
  renderRoleFitHtml({
    organisation: 'Acme Pty Ltd', team: 'Sales Team', numMembers: 15, dateStr: '2026-11-19',
    occupation: { title: occupation.title, profileDate: occupation.profileDate }, profile, candidates, ...over,
  });
const visible = (html: string) => html.replace(/<style>[\s\S]*?<\/style>/g, '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');

describe('Role Fit report: cover and overview', () => {
  it('shows the organisation, team, number of team members and date on the cover', () => {
    const text = visible(render());
    expect(text).toContain('Organisation: Acme Pty Ltd');
    expect(text).toContain('Team: Sales Team');
    expect(text).toContain('Number of Team members: 15');
    expect(text).toContain('Date: 19 November 2026');
  });

  it('carries the title, the copyright and O*NET trademark notice, and the overview text from the owner\'s template', () => {
    const text = visible(render());
    expect(text).toContain('SSM® Role Fit | Interpretative Report');
    expect(text).toContain('All rights reserved');
    expect(text).toContain('O*NET is a trademark');
    expect(text).toContain('About This Report');
    expect(text).toContain('Role Fit Analysis'); // the contents list
  });

  it('prints the team name in the page header of every following page', () => {
    expect(render().match(/Sales Team \| Page \d/g)).toHaveLength(4);
  });

  it('escapes organisation and team names, which a buyer types freely', () => {
    const html = render({ organisation: '<script>x</script>', team: '"><img src=x>' });
    expect(html).not.toContain('<script>x</script>');
    expect(html).not.toContain('"><img src=x>');
  });
});

describe('Role Fit report: target role profile', () => {
  it('names the occupation and the O*NET profile date, which is not the report date', () => {
    const text = visible(render());
    expect(text).toContain('Occupation: Chief Executives');
    expect(text).toContain('ONET profile date: 12/2025');
  });

  it('omits the profile date line cleanly when O*NET gave none', () => {
    const text = visible(render({ occupation: { title: 'Chief Executives' } }));
    expect(text).toContain('Occupation: Chief Executives');
    expect(text).not.toContain('ONET profile date');
    expect(text).not.toContain('undefined');
  });

  it('lists all 12 strengths with rank, category and definition, in the role\'s order', () => {
    const html = render();
    const pageStart = html.indexOf('Target Role Profile');
    const page = html.slice(pageStart, html.indexOf('Role-Fit Analysis', pageStart));
    let last = -1;
    for (const t of TRAITS) {
      const at = page.indexOf(`<b>${t}</b>`);
      expect(at).toBeGreaterThan(last);
      last = at;
    }
    expect(page).toContain('Signature');
    expect(page).toContain('Situational');
    expect(page).toContain('The capacity to');
  });
});

describe('Role Fit report: candidate analysis', () => {
  it('lists candidates in rank order with their stars: the matching candidate has 5, the exact opposite 1', () => {
    const html = render();
    expect(html.indexOf('Alice Fit')).toBeLessThan(html.indexOf('Bob Far'));
    expect([...html.matchAll(/data-stars="(\d)"/g)].map((m) => Number(m[1]))).toEqual([5, 1]);
  });

  it('shows five different star levels, best first, from the fixed quintile cut-offs', () => {
    // Fit falls step by step: a perfect match, the role's top strength ranked last, then the top 8 and top 9
    // strengths reversed, then the exact opposite of the role.
    const reverseFirst = (n: number) => [...TRAITS.slice(0, n).reverse(), ...TRAITS.slice(n)];
    const orders = [TRAITS, [...TRAITS.slice(1), TRAITS[0]], reverseFirst(8), reverseFirst(9), [...TRAITS].reverse()];
    const five = rankCandidates(orders.map((order, i) => ({ id: String(i), name: `Cand ${i}`, ranks: ranksOf(order) })), profile);
    const html = render({ candidates: five });
    expect([...html.matchAll(/data-stars="(\d)"/g)].map((m) => Number(m[1]))).toEqual([5, 4, 3, 2, 1]);
  });

  it('shows each candidate\'s aligned and misaligned strengths, comma separated', () => {
    const html = render();
    const alice = candidates[0];
    expect(visible(html)).toContain(alice.aligned.join(', '));
    const bob = candidates[1];
    expect(visible(html)).toContain(bob.misaligned.join(', '));
  });

  it('does not print any candidate\'s own fit percentage: the template shows stars only', () => {
    const text = visible(render());
    // Whole numbers only: the legend legitimately contains "20%", which contains "0%".
    const printed = (value: string) => new RegExp(`(?<![\\d.])${value.replace('.', '\\.')}%`);
    for (const c of candidates) {
      expect(text).not.toMatch(printed(c.score.toFixed(1)));
      expect(text).not.toMatch(printed(String(Math.round(c.score))));
    }
    expect(text).not.toMatch(/Very (Strong|Weak) Fit/);
  });

  it('explains stars as the client\'s template does: the top 20% of all people for 5 stars, the bottom 20% for 1', () => {
    const text = visible(render());
    expect(text).toContain("5 stars \u2014 The candidate's Strength fit is in the top 20% of all people.");
    expect(text).toContain("1 star \u2014 The candidate's Strength fit is in the bottom 20% of all people.");
    expect(text).toContain('21st\u201340th percentile');
    expect(text).not.toContain('candidates in this report');
  });

  it('escapes candidate names', () => {
    const evil = rankCandidates([{ id: '1', name: '<b onload=x>Eve', ranks: ranksOf(TRAITS) }], profile);
    expect(render({ candidates: evil })).not.toContain('<b onload=x>');
  });

  it('copes with a single candidate and with none', () => {
    expect(render({ candidates: candidates.slice(0, 1) })).toContain('Alice Fit');
    expect(() => render({ candidates: [] })).not.toThrow();
  });
});

describe('roleFitFilename', () => {
  it('is ASCII-only for any organisation name', () => {
    expect(roleFitFilename('Zoë & Müller/Co', '2026-11-19')).toMatch(/^[A-Za-z0-9._-]+$/);
  });
});

describe('Role Fit report: page furniture the buyer relies on', () => {
  it('separates the four categories in the Target Role Profile table', () => {
    const pages = render().split('<section class="page">');
    const profilePage = pages.find((p) => p.includes('Occupation: Chief Executives'))!;
    expect([...profilePage.matchAll(/class="spacer"/g)]).toHaveLength(3);
  });

  it('shows the ShawSight logo, each strength\'s definition, and the note that strengths are weighted by importance', () => {
    const html = render();
    expect(html).toContain('alt="ShawSight"');
    expect(visible(html)).toContain('The capacity to consistently execute planned actions');
    expect(visible(html)).toContain('Not every strength matters equally');
  });
});
