import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { individualCopy } from '../copy/index.js';
import {
  categoryForRankNumber,
  colorForCategory,
  formatRank,
} from '../scoring/traits.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOGO_PATH = join(__dirname, '../../logo.png');

function logoDataUri(): string {
  if (!existsSync(LOGO_PATH)) return '';
  const b64 = readFileSync(LOGO_PATH).toString('base64');
  return `data:image/png;base64,${b64}`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatLongDate(dateStr: string): string {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

type RankRow = { rank: number; category: string; trait: string; color: string };

function buildRows(
  orderedTraits: string[],
  ranks: Record<string, number>,
): RankRow[] {
  return orderedTraits.map((trait) => {
    const rank = ranks[trait];
    const category = categoryForRankNumber(rank);
    return { rank, category, trait, color: colorForCategory(category) };
  });
}

function rowsHtml(rows: RankRow[], extraCell: (row: RankRow) => string): string {
  let html = '';
  let lastCat: string | null = null;
  for (const row of rows) {
    if (lastCat && lastCat !== row.category) {
      html += `<tr class="spacer"><td colspan="10"></td></tr>`;
    }
    lastCat = row.category;
    html += `<tr>
      <td class="center cat" style="background:${row.color}">${formatRank(row.rank)}</td>
      <td class="center cat" style="background:${row.color}">${row.category}</td>
      <td class="center"><b>${escapeHtml(row.trait)}</b></td>
      ${extraCell(row)}
    </tr>`;
  }
  return html;
}

const BASE_CSS = `
  * { box-sizing: border-box; }
  body { font-family: Helvetica, Arial, sans-serif; font-size: 11px; color: #111; margin: 0; }
  h1 { font-size: 22px; margin: 0 0 8px; }
  h2 { font-size: 20px; margin: 0; }
  .page { page-break-after: always; }
  .page:last-child { page-break-after: auto; }
  .gold-bar { height: 10px; background: #ba9354; margin: 8px 0 12px; }
  .box { border: 1px solid #000; padding: 10px; margin-bottom: 10px; }
  .footer-row { display: flex; align-items: center; gap: 16px; margin-top: 240px; }
  .footer-row img { width: 110px; height: auto; }
  .hdr { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; }
  table.results { width: 100%; border-collapse: collapse; font-size: 9px; }
  table.results th, table.results td { border: 1px solid #000; padding: 4px 5px; vertical-align: middle; }
  table.results th { background: #f3f3f3; }
  .center { text-align: center; }
  tr.spacer td { border: none; height: 6px; padding: 0; }
  .info { border: 1px solid #000; padding: 10px; font-size: 9.5px; line-height: 1.35; margin-top: 12px; }
  .matrix { width: 100%; border-collapse: collapse; text-align: center; font-weight: bold; font-size: 10px; }
  .matrix td { border: 1px solid #000; padding: 8px 4px; }
  .matrix .cell-blue { background: #4d93d9; }
`;

export function renderIndividualHtml(opts: {
  first: string;
  last: string;
  dateStr: string;
  orderedTraits: string[];
  ranks: Record<string, number>;
}): string {
  const copy = individualCopy;
  const name = `${opts.first} ${opts.last}`;
  const dateLong = formatLongDate(opts.dateStr);
  const logo = logoDataUri();
  const rows = buildRows(opts.orderedTraits, opts.ranks);

  const assessmentTable = rowsHtml(
    rows,
    (row) =>
      `<td>${copy.descriptions[row.trait as keyof typeof copy.descriptions] ?? ''}</td>`,
  );

  const stylesTable = rowsHtml(rows, (row) => {
    const style = copy.onetStyles[row.trait as keyof typeof copy.onetStyles];
    return `<td class="center"><b>${style?.label ?? ''}</b></td><td>${style?.description ?? ''}</td>`;
  });

  const activitiesTable = rowsHtml(
    rows,
    (row) =>
      `<td class="center">${copy.onetActivities[row.trait as keyof typeof copy.onetActivities] ?? ''}</td>`,
  );

  const header = (page: number, subtitle: string) => `
    <div class="hdr">
      <div>Shaw Strengths Matrix® Assessment</div>
      <div>${escapeHtml(name)} | Page ${page}</div>
    </div>
    <h2>${subtitle}</h2>
    <div style="height:8px"></div>`;

  const matrix = `
    <table class="matrix">
      <tr><td colspan="5">Shaw Strengths Matrix®</td></tr>
      <tr><td colspan="2" rowspan="2">Character Strengths</td><td colspan="3">Temporal Preferences</td></tr>
      <tr><td>Past Reflection</td><td>Present Engagement</td><td>Future Anticipation</td></tr>
      <tr><td rowspan="4">Cognitive Preferences</td><td>Intuition</td><td class="cell-blue">Foresight</td><td class="cell-blue">Confidence</td><td class="cell-blue">Courage</td></tr>
      <tr><td>Thinking</td><td class="cell-blue">Curiosity</td><td class="cell-blue">Objectivity</td><td class="cell-blue">Fairness</td></tr>
      <tr><td>Feeling</td><td class="cell-blue">Empathy</td><td class="cell-blue">Tenacity</td><td class="cell-blue">Prudence</td></tr>
      <tr><td>Sensing</td><td class="cell-blue">Discernment</td><td class="cell-blue">Practicality</td><td class="cell-blue">Discipline</td></tr>
    </table>`;

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>${BASE_CSS}</style></head>
<body>
  <section class="page">
    <h1>Shaw Strengths Matrix® Assessment</h1>
    <div class="gold-bar"></div>
    <div class="box">
      <div><b>Report prepared for</b></div>
      <div><b>Name: ${escapeHtml(name)}</b></div>
      <div><b>Date: ${escapeHtml(dateLong)}</b></div>
    </div>
    <div class="footer-row">
      ${logo ? `<img src="${logo}" alt="ShawSight"/>` : ''}
      <div>ShawSight Pty Ltd &nbsp;|&nbsp; ABN: 38688414557</div>
    </div>
    <div class="box" style="margin-top:12px">${copy.legal}</div>
  </section>

  <section class="page">
    ${header(2, 'Overview')}
    <div class="box">${copy.overview}</div>
    <div class="box">${copy.contents}</div>
  </section>

  <section class="page">
    ${header(3, 'Shaw Strengths Matrix®')}
    ${matrix}
    <div class="info">${copy.howToReadMatrix}</div>
  </section>

  <section class="page">
    ${header(4, 'Assessment Table')}
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Description</th></tr></thead>
      <tbody>${assessmentTable}</tbody>
    </table>
    <div class="info">${copy.howToReadAssessmentTable}</div>
  </section>

  <section class="page">
    ${header(5, 'Mapping to O*NET® Work Styles')}
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Work Style (O*NET®)</th><th>Description</th></tr></thead>
      <tbody>${stylesTable}</tbody>
    </table>
    <div class="info">${copy.howToReadWorkStyles}</div>
  </section>

  <section class="page">
    ${header(6, 'Mapping to O*NET® Work Activities')}
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Work Activities (O*NET®)</th></tr></thead>
      <tbody>${activitiesTable}</tbody>
    </table>
  </section>

  <section class="page">
    ${header(7, 'Mapping to O*NET® Work Activities')}
    <div class="info">${copy.howToReadWorkActivities}</div>
  </section>
</body></html>`;
}

export function individualFilename(
  first: string,
  last: string,
  dateStr: string,
): string {
  return `SSM_${first}_${last}_${dateStr}_v1.pdf`;
}
