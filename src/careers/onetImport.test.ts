import { describe, it, expect } from 'vitest';
import { TRAITS } from '../scoring/traits.js';
import { buildOccupations, parseCsv } from './onetImport.js';

const mapping = Object.fromEntries(TRAITS.map((t) => [t, `Style ${t}`]));

const WS_HEADER =
  'O*NET-SOC Code,Title,Element ID,Element Name,Scale ID,Scale Name,Data Value,Date,Domain Source';
const OI_HEADER = WS_HEADER;

function workStyleRows(code: string, title: string, impact: number, skip?: string) {
  return TRAITS.filter((t) => t !== skip).flatMap((t) => [
    `${code},"${title}",1.D.1,Style ${t},WI,Work Styles Impact,${impact},12/2025,AI/Expert`,
    `${code},"${title}",1.D.1,Style ${t},DR,Distinctiveness Rank,0,12/2025,AI/Expert`,
  ]);
}

function interestRows(code: string, title: string, skipArea?: string) {
  const areas = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional'];
  return [
    ...areas
      .filter((a) => a !== skipArea)
      .map((a, i) => `${code},"${title}",1.B.1,${a},OI,Occupational Interests,${i + 1},02/2026,ML`),
    `${code},"${title}",1.B.1.g,First Interest High-Point,IH,High-Point,3,02/2026,ML`,
  ];
}

describe('parseCsv', () => {
  it('keeps commas inside quoted fields (occupation titles contain them)', () => {
    const rows = parseCsv('a,b\n"Managers, All Other",x');
    expect(rows).toEqual([{ a: 'Managers, All Other', b: 'x' }]);
  });

  it('handles a BOM and CRLF line endings', () => {
    expect(parseCsv('﻿a,b\r\n1,2\r\n')).toEqual([{ a: '1', b: '2' }]);
  });
});

describe('buildOccupations', () => {
  it('turns Work Styles Impact (-3..3) into a 0-100 requirement per SSM strength', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('11-1011.00', 'CEO', 3)].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('11-1011.00', 'CEO')].join('\n'));
    const [occ] = buildOccupations(ws, oi, mapping);
    expect(occ.requirements.Initiative).toBe(100);
  });

  it('treats negative impact as 0: a style the job does not want is not a requirement', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('11-1011.00', 'CEO', -1.5)].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('11-1011.00', 'CEO')].join('\n'));
    expect(buildOccupations(ws, oi, mapping)[0].requirements.Empathy).toBe(0);
  });

  it('reads RIASEC from Occupational Interests, ignoring the high-point rank rows', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('11-1011.00', 'CEO', 1)].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('11-1011.00', 'CEO')].join('\n'));
    expect(buildOccupations(ws, oi, mapping)[0].riasec).toEqual({ R: 1, I: 2, A: 3, S: 4, E: 5, C: 6 });
  });

  it('drops occupations missing any mapped work style or any interest area, rather than scoring on partial data', () => {
    const ws = parseCsv(
      [WS_HEADER, ...workStyleRows('A', 'Full', 1), ...workStyleRows('B', 'NoStyle', 1, 'Initiative'), ...workStyleRows('C', 'NoInterest', 1)].join('\n'),
    );
    const oi = parseCsv(
      [OI_HEADER, ...interestRows('A', 'Full'), ...interestRows('B', 'NoStyle'), ...interestRows('C', 'NoInterest', 'Social')].join('\n'),
    );
    expect(buildOccupations(ws, oi, mapping).map((o) => o.code)).toEqual(['A']);
  });

  it('fails loudly when the mapping names a work style O*NET does not have, instead of silently scoring zero', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('A', 'Full', 1)].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('A', 'Full')].join('\n'));
    expect(() => buildOccupations(ws, oi, { ...mapping, Initiative: 'Not A Real Style' })).toThrow(/Not A Real Style/);
  });
});

describe('parseCsv: edge cases the real O*NET files can contain', () => {
  it('reads a doubled quote inside a quoted field as one quote character', () => {
    expect(parseCsv('a\n"He said ""hi"""')).toEqual([{ a: 'He said "hi"' }]);
  });

  it('counts a CRLF as one line break, so Windows files do not gain a blank row between records', () => {
    expect(parseCsv('a,b\r\n1,2\r\n3,4')).toEqual([{ a: '1', b: '2' }, { a: '3', b: '4' }]);
  });

  it('keeps a newline that sits inside quotes as part of the field', () => {
    expect(parseCsv('a,b\n"line1\nline2",x')).toEqual([{ a: 'line1\nline2', b: 'x' }]);
  });

  it('skips blank lines and keeps the last record when the file has no trailing newline', () => {
    expect(parseCsv('a,b\n\n1,2\n\n3,4')).toEqual([{ a: '1', b: '2' }, { a: '3', b: '4' }]);
  });

  it('keeps empty values and fills a short row with empty strings', () => {
    expect(parseCsv('a,b,c\n1,,3\n4')).toEqual([{ a: '1', b: '', c: '3' }, { a: '4', b: '', c: '' }]);
  });

  it('strips a BOM only at the very start of the file', () => {
    expect(parseCsv('a\nx﻿y')).toEqual([{ a: 'x﻿y' }]);
  });
});

describe('buildOccupations: what ends up in the generated data file', () => {
  const build = (codes: string[], extraOi: string[] = [], impact = 1) => {
    const ws = parseCsv([WS_HEADER, ...codes.flatMap((c) => workStyleRows(c, `Title ${c}`, impact))].join('\n'));
    const oi = parseCsv([OI_HEADER, ...codes.flatMap((c) => interestRows(c, `Title ${c}`)), ...extraOi].join('\n'));
    return buildOccupations(ws, oi, mapping);
  };

  it('returns occupations in code order whatever the file order, with their titles, so the data file is stable', () => {
    expect(build(['C-3', 'A-1', 'B-2']).map((o) => [o.code, o.title])).toEqual([['A-1', 'Title A-1'], ['B-2', 'Title B-2'], ['C-3', 'Title C-3']]);
  });

  it('ignores interest rows that are not Occupational Interests, even when they carry an area name', () => {
    const extra = ['A,"X",1.B.1,Realistic,IH,High-Point,99,02/2026,ML', 'A,"X",1.B.1,Unrelated,OI,Occupational Interests,99,02/2026,ML'];
    expect(build(['A'], extra)[0].riasec).toEqual({ R: 1, I: 2, A: 3, S: 4, E: 5, C: 6 });
  });

  it('takes the requirement from Work Styles Impact only, not from the distinctiveness-rank rows (which carry 0 here)', () => {
    expect(build(['A'], [], 1.5)[0].requirements.Empathy).toBe(50);
  });
});

describe('buildOccupations: profile date', () => {
  it('carries the date O*NET recorded for the occupation\'s work styles, which the Role Fit report prints', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('A', 'X', 1)].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('A', 'X')].join('\n'));
    expect(buildOccupations(ws, oi, mapping)[0].profileDate).toBe('12/2025');
  });

  it('leaves the date out, rather than writing an empty string, when O*NET gave none', () => {
    const ws = parseCsv([WS_HEADER, ...workStyleRows('A', 'X', 1).map((l) => l.replace(',12/2025,', ',,'))].join('\n'));
    const oi = parseCsv([OI_HEADER, ...interestRows('A', 'X')].join('\n'));
    expect(buildOccupations(ws, oi, mapping)[0].profileDate).toBeUndefined();
  });
});
