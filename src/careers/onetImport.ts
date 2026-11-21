import { TRAITS, type Trait } from '../scoring/traits.js';
import { WORK_STYLE_IMPACT_MAX } from './constants.js';
import type { Occupation, Riasec } from './types.js';

/** Minimal RFC-4180 CSV reader (quoted fields, doubled quotes, BOM, CRLF). Returns one object per row. */
export function parseCsv(text: string): Array<Record<string, string>> {
  const src = text.replace(/^﻿/, '');
  const records: string[][] = [];
  let field = '';
  let record: string[] = [];
  let quoted = false;
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (quoted) {
      if (ch === '"' && src[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      record.push(field);
      field = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && src[i + 1] === '\n') i++;
      record.push(field);
      field = '';
      if (record.some((f) => f !== '')) records.push(record);
      record = [];
    } else {
      field += ch;
    }
  }
  if (field !== '' || record.length) {
    record.push(field);
    if (record.some((f) => f !== '')) records.push(record);
  }
  const [header, ...rows] = records;
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}

const AREA_BY_NAME: Record<string, keyof Riasec> = {
  Realistic: 'R',
  Investigative: 'I',
  Artistic: 'A',
  Social: 'S',
  Enterprising: 'E',
  Conventional: 'C',
};

const CODE = 'O*NET-SOC Code';

/**
 * Join O*NET work_styles.csv and career_interest_types.csv rows into SSM-shaped occupations.
 * `strengthToWorkStyle` is the owner's SSM -> O*NET Work Style mapping (the other work styles are dropped).
 */
export function buildOccupations(
  workStyleRows: Array<Record<string, string>>,
  interestRows: Array<Record<string, string>>,
  strengthToWorkStyle: Record<string, string>,
): Occupation[] {
  const known = new Set(workStyleRows.map((r) => r['Element Name']));
  for (const trait of TRAITS) {
    const style = strengthToWorkStyle[trait];
    if (!style || !known.has(style)) {
      throw new Error(`Mapping for ${trait} names work style "${style}", which is not in the O*NET data`);
    }
  }

  const styleByOcc = new Map<string, { title: string; date: string; impact: Map<string, number> }>();
  for (const r of workStyleRows) {
    if (r['Scale ID'] !== 'WI') continue;
    const entry = styleByOcc.get(r[CODE]) ?? { title: r['Title'], date: r['Date'], impact: new Map<string, number>() };
    entry.impact.set(r['Element Name'], Number(r['Data Value']));
    styleByOcc.set(r[CODE], entry);
  }

  const interestByOcc = new Map<string, Partial<Riasec>>();
  for (const r of interestRows) {
    const area = AREA_BY_NAME[r['Element Name']];
    if (r['Scale ID'] !== 'OI' || !area) continue;
    const entry = interestByOcc.get(r[CODE]) ?? {};
    entry[area] = Number(r['Data Value']);
    interestByOcc.set(r[CODE], entry);
  }

  const occupations: Occupation[] = [];
  for (const [code, { title, date, impact }] of styleByOcc) {
    const interests = interestByOcc.get(code);
    if (!interests || Object.keys(interests).length !== 6) continue;
    const requirements = {} as Record<Trait, number>;
    let complete = true;
    for (const trait of TRAITS) {
      const value = impact.get(strengthToWorkStyle[trait]);
      if (value === undefined) {
        complete = false;
        break;
      }
      requirements[trait] =
        Math.round((Math.max(0, value) / WORK_STYLE_IMPACT_MAX) * 1000) / 10;
    }
    if (!complete) continue;
    occupations.push({ code, title, profileDate: date || undefined, riasec: interests as Riasec, requirements });
  }
  return occupations.sort((a, b) => a.code.localeCompare(b.code));
}
