import { careerCopy, individualCopy } from '../copy/index.js';
import { STAR_LEVELS } from '../careers/constants.js';
import type { RatedOccupation, Riasec } from '../careers/types.js';
import { escapeHtml } from '../html.js';
import { fileSafe } from '../filenames.js';
import {
  BASE_CSS,
  buildRows,
  formatLongDate,
  logoDataUri,
  rowsHtml,
} from './individual.js';

const CAREER_CSS = `
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 10.5px; }
  .bar-label { width: 90px; }
  .bar-track { flex: 1; background: #eee; height: 12px; }
  .bar-fill { background: #4d93d9; height: 12px; }
  table.matches { width: 100%; border-collapse: collapse; font-size: 11px; }
  table.matches td { border-bottom: 1px solid #ddd; padding: 5px 4px; }
  table.matches td.rank { width: 34px; font-weight: bold; color: #555; }
  table.matches td.stars { width: 110px; white-space: nowrap; }
  table.matches tr { page-break-inside: avoid; }
  .attribution { font-size: 8.5px; margin-top: 14px; color: #444; }
  .archetype { border: 2px solid #4d93d9; background: #eef6fd; padding: 12px; margin: 12px 0; font-size: 11.5px; line-height: 1.5; }
  .archetype h3 { margin: 0 0 6px; font-size: 14px; }
`;

const star = (fill: string) =>
  `<svg width="14" height="14" viewBox="0 0 24 24" style="vertical-align:middle"><polygon fill="${fill}" points="12,1.5 15.1,8.6 22.8,9.3 17,14.4 18.8,22 12,18 5.2,22 7,14.4 1.2,9.3 8.9,8.6"/></svg>`;

export function starsHtml(n: number): string {
  return `<span data-stars="${n}">${star('#ba9354').repeat(n)}${star('#d9d9d9').repeat(STAR_LEVELS - n)}</span>`;
}

function interestBars(riasec: Riasec): string {
  const max = Math.max(...Object.values(riasec)) || 1;
  return (Object.keys(careerCopy.areas) as Array<keyof Riasec>)
    .map(
      (a) => `<div class="bar-row"><div class="bar-label">${careerCopy.areas[a]}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.round((riasec[a] / max) * 100)}%"></div></div></div>`,
    )
    .join('');
}

/** "A, B and C" */
function joinList(items: string[]): string {
  return items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/**
 * One-sentence summary: the user's two strongest interest areas and their top three strengths.
 * A user with no interest preference (all six scores equal) gets the strengths clause only.
 */
export function summarySentence(riasec: Riasec, topStrengths: string[]): string {
  const areas = (Object.keys(careerCopy.areas) as Array<keyof Riasec>)
    .map((a, i) => ({ name: careerCopy.areas[a], score: riasec[a], i }))
    .sort((a, b) => b.score - a.score || a.i - b.i);
  const hasPreference = areas[0].score !== areas[areas.length - 1].score;
  const strengths = joinList(topStrengths);
  return hasPreference
    ? careerCopy.summaryBoth.replace('{areas}', joinList(areas.slice(0, 2).map((a) => a.name))).replace('{strengths}', strengths)
    : careerCopy.summaryStrengthsOnly.replace('{strengths}', strengths);
}

/** Paid report. Occupations are shown as stars only; within a star level they are alphabetical. */
export function renderCareerHtml(opts: {
  first: string;
  last: string;
  dateStr: string;
  orderedTraits: string[];
  ranks: Record<string, number>;
  riasec: Riasec;
  matches: RatedOccupation[];
  /** One Work Archetype per strength, best-ranked first. */
  archetypes: Array<{ trait: string; text: string }>;
  attribution: string;
}): string {
  const copy = careerCopy;
  const name = `${opts.first} ${opts.last}`;
  const logo = logoDataUri();
  const strengths = rowsHtml(
    buildRows(opts.orderedTraits, opts.ranks),
    (row) =>
      `<td>${individualCopy.descriptions[row.trait as keyof typeof individualCopy.descriptions] ?? ''}</td>`,
  );
  const matchRows = [...opts.matches]
    .sort((a, b) => a.rank - b.rank)
    .map(
      (m) =>
        `<tr><td class="rank">${m.rank}</td><td class="stars">${starsHtml(m.stars)}</td><td>${escapeHtml(m.title)}</td></tr>`,
    )
    .join('');
  const summary = summarySentence(opts.riasec, opts.orderedTraits.slice(0, 3));

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>${BASE_CSS}${CAREER_CSS}</style></head>
<body>
  <section class="page">
    <h1>${copy.reportTitle}</h1>
    <div class="gold-bar"></div>
    <div class="box">
      <div><b>Prepared for: ${escapeHtml(name)}</b></div>
      <div><b>Date: ${escapeHtml(formatLongDate(opts.dateStr))}</b></div>
    </div>
    <p>${copy.intro}</p>

    <div class="archetype">
      <h3>${copy.archetypeHeading}</h3>
      <p>${copy.archetypeIntro}</p>
      ${opts.archetypes
        .map((a, i) => `<p><b>${i + 1}. ${escapeHtml(a.trait)}:</b> ${escapeHtml(a.text)}</p>`)
        .join('')}
    </div>

    <h2>Your Ranked Strengths</h2>
    <div style="height:6px"></div>
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Description</th></tr></thead>
      <tbody>${strengths}</tbody>
    </table>

    <h2 style="margin-top:16px">${copy.interestsHeading}</h2>
    <p style="font-size:10px">${copy.interestsNote}</p>
    ${interestBars(opts.riasec)}
  </section>

  <section class="page">
    <h2>${copy.matchesHeading}</h2>
    <p style="font-size:10px">${copy.matchesNote}</p>
    <table class="matches"><tbody>${matchRows}</tbody></table>
    <p><b>${escapeHtml(summary)}</b></p>
    <div class="info">${copy.fitNotCompetence.join('<br/>')}</div>
    <div class="attribution">${escapeHtml(opts.attribution)}<br/>${copy.attribution}</div>
    <div class="footer-row" style="margin-top:20px">
      ${logo ? `<img src="${logo}" alt="ShawSight"/>` : ''}
      <div>ShawSight Pty Ltd &nbsp;|&nbsp; ABN: 38688414557</div>
    </div>
  </section>
</body></html>`;
}

export function careerFilename(first: string, last: string, dateStr: string): string {
  return `SSM_Career_Matches_${fileSafe(first)}_${fileSafe(last)}_${dateStr}.pdf`;
}
