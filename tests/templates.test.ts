import { describe, expect, it } from 'vitest';
import { TRAITS } from '../src/scoring/traits.js';
import { renderIndividualHtml } from '../src/templates/individual.js';
import { renderTeamHtml } from '../src/templates/team.js';

describe('HTML templates', () => {
  it('renders individual report shell with ranked traits', () => {
    const ranks = Object.fromEntries(TRAITS.map((t, i) => [t, i + 1]));
    const html = renderIndividualHtml({
      first: 'Ada',
      last: 'Lovelace',
      dateStr: '2026-09-14',
      orderedTraits: [...TRAITS],
      ranks,
    });
    expect(html).toContain('Ada Lovelace');
    expect(html).toContain('Assessment Table');
    expect(html).toContain('Courage');
    expect(html).toContain('Signature');
  });

  it('renders team report with distribution chart svg', () => {
    const ranks = Object.fromEntries(TRAITS.map((t, i) => [t, i + 1]));
    const distributionData = Object.fromEntries(
      TRAITS.map((t) => [
        t,
        { Signature: 0.25, Supporting: 0.25, Stretch: 0.25, Situational: 0.25 },
      ]),
    );
    const html = renderTeamHtml({
      companyName: 'Acme',
      teamName: 'Alpha',
      numMembers: 4,
      dateStr: '2026-09-14',
      orderedTraits: [...TRAITS],
      ranks,
      distributionData: distributionData as any,
    });
    expect(html).toContain('Acme');
    expect(html).toContain('Team Strength Distribution');
    expect(html).toContain('<svg');
    expect(html).toContain('Change Alignment Guide');
  });
});
