import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { individualCopy, teamCopy } from '../copy/index.js';
import {
  categoryForRankNumber,
  colorForCategory,
  formatRank,
  type Category,
} from '../scoring/traits.js';
import type { DistributionData } from '../scoring/teamRankings.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOGO_PATH = join(__dirname, '../../logo.png');

function logoDataUri(): string {
  if (!existsSync(LOGO_PATH)) return '';
  return `data:image/png;base64,${readFileSync(LOGO_PATH).toString('base64')}`;
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

function distributionChartSvg(
  orderedTraits: string[],
  distributionData: DistributionData,
): string {
  const cats: Category[] = ['Signature', 'Supporting', 'Stretch', 'Situational'];
  const colors = cats.map((c) => colorForCategory(c));
  const width = 700;
  const height = 320;
  const left = 50;
  const bottom = 50;
  const chartW = width - 100;
  const chartH = height - 100;
  const n = orderedTraits.length;
  const barW = chartW / (n * 1.5);
  const gap = barW * 0.5;

  let bars = '';
  orderedTraits.forEach((trait, i) => {
    const x = left + i * (barW + gap);
    let y = bottom + chartH;
    for (let c = 0; c < cats.length; c++) {
      const pct = distributionData[trait]?.[cats[c]] ?? 0;
      const h = pct * chartH;
      y -= h;
      if (h > 0) {
        bars += `<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${colors[c]}" stroke="#000" stroke-width="0.5"/>`;
      }
    }
    const label = trait.slice(0, 4);
    bars += `<text x="${x + barW / 2}" y="${bottom + chartH + 14}" text-anchor="middle" font-size="8" transform="rotate(-40 ${x + barW / 2} ${bottom + chartH + 14})">${escapeHtml(label)}</text>`;
  });

  let grid = '';
  for (let i = 0; i <= 100; i += 20) {
    const y = bottom + chartH - (i / 100) * chartH;
    grid += `<line x1="${left}" y1="${y}" x2="${left + chartW}" y2="${y}" stroke="#ccc" stroke-width="0.5"/>`;
    grid += `<text x="${left - 6}" y="${y + 3}" text-anchor="end" font-size="8">${i}%</text>`;
  }

  let legend = '';
  const legendY = 18;
  let lx = left;
  const labels = [
    'Signature (#1-#3)',
    'Supporting (#4-#6)',
    'Stretch (#7-#9)',
    'Situational (#10-#12)',
  ];
  cats.forEach((c, i) => {
    legend += `<rect x="${lx}" y="${legendY}" width="12" height="12" fill="${colors[i]}" stroke="#000"/>`;
    legend += `<text x="${lx + 16}" y="${legendY + 10}" font-size="9">${labels[i]}</text>`;
    lx += 150;
  });

  return `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${legend}${grid}${bars}</svg>`;
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
  .footer-row { display: flex; align-items: center; gap: 16px; margin-top: 200px; }
  .footer-row img { width: 110px; height: auto; }
  .hdr { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 11px; }
  table.results { width: 100%; border-collapse: collapse; font-size: 8px; }
  table.results th, table.results td { border: 1px solid #000; padding: 3px 4px; vertical-align: middle; }
  .center { text-align: center; }
  tr.spacer td { border: none; height: 6px; padding: 0; }
  .info { border: 1px solid #000; padding: 10px; font-size: 9.5px; line-height: 1.35; margin-top: 12px; }
  .matrix { width: 100%; border-collapse: collapse; text-align: center; font-weight: bold; font-size: 10px; }
  .matrix td { border: 1px solid #000; padding: 8px 4px; }
  .matrix .cell-blue { background: #4d93d9; }
  .chart-wrap { text-align: center; margin: 8px 0; }
`;

export function renderTeamHtml(opts: {
  companyName: string;
  teamName: string;
  numMembers: number;
  dateStr: string;
  orderedTraits: string[];
  ranks: Record<string, number>;
  distributionData: DistributionData;
}): string {
  const copy = teamCopy;
  const ind = individualCopy;
  const dateLong = formatLongDate(opts.dateStr);
  const logo = logoDataUri();
  const rows = buildRows(opts.orderedTraits, opts.ranks);

  const assessmentTable = rowsHtml(
    rows,
    (row) =>
      `<td>${ind.descriptions[row.trait as keyof typeof ind.descriptions] ?? ''}</td>`,
  );

  const stylesTable = rowsHtml(rows, (row) => {
    const style = ind.onetStyles[row.trait as keyof typeof ind.onetStyles];
    return `<td class="center"><b>${style?.label ?? ''}</b></td><td>${style?.description ?? ''}</td>`;
  });

  const activitiesTable = rowsHtml(
    rows,
    (row) =>
      `<td class="center">${ind.onetActivities[row.trait as keyof typeof ind.onetActivities] ?? ''}</td>`,
  );

  let changeRows = '';
  let lastCat: string | null = null;
  for (const row of rows) {
    if (lastCat && lastCat !== row.category) {
      changeRows += `<tr class="spacer"><td colspan="5"></td></tr>`;
    }
    lastCat = row.category;
    const response =
      copy.strengthResponse[row.trait as keyof typeof copy.strengthResponse] ??
      row.trait;
    const rr = copy.riskReframe[row.trait as keyof typeof copy.riskReframe];
    changeRows += `<tr>
      <td class="center" style="background:${row.color}">${formatRank(row.rank)}</td>
      <td class="center" style="background:${row.color}">${row.category}</td>
      <td class="center">${response}</td>
      <td class="center">${rr?.risk ?? ''}</td>
      <td class="center">${rr?.reframe ?? ''}</td>
    </tr>`;
  }

  const header = (page: number, subtitle: string) => `
    <div class="hdr">
      <div>Shaw Strengths Matrix® Assessment</div>
      <div>${escapeHtml(opts.teamName)} | Page ${page}</div>
    </div>
    <h2>${subtitle}</h2>
    <div style="height:8px"></div>`;

  const matrix = `
    <table class="matrix">
      <tr><td colspan="5">Shaw Strengths Matrix®</td></tr>
      <tr><td colspan="2" rowspan="2">Character Strengths</td><td colspan="3">Temporal Preferences</td></tr>
      <tr><td>Past Reflection</td><td>Present Engagement</td><td>Future Anticipation</td></tr>
      <tr><td rowspan="4">Cognitive Preferences</td><td>Intuition</td><td class="cell-blue">Foresight</td><td class="cell-blue">Confidence</td><td class="cell-blue">Initiative</td></tr>
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
      <div><b>Organisation: ${escapeHtml(opts.companyName)}</b></div>
      <div><b>Team: ${escapeHtml(opts.teamName)}</b></div>
      <div><b>Number of Team members: ${opts.numMembers}</b></div>
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
    ${header(4, 'Team Assessment Table')}
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Description</th></tr></thead>
      <tbody>${assessmentTable}</tbody>
    </table>
    <div class="info">${copy.howToReadAssessmentTable}</div>
  </section>

  <section class="page">
    ${header(5, 'Team Mapping to O*NET® Work Styles')}
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
    <div class="info">${copy.howToReadWorkActivities}</div>
  </section>

  <section class="page">
    ${header(8, 'Team Strength Distribution')}
    <div class="chart-wrap">${distributionChartSvg(opts.orderedTraits, opts.distributionData)}</div>
    <div class="info">${copy.howToReadDistribution}</div>
  </section>

  <section class="page">
    ${header(9, 'Change Alignment Guide')}
    <table class="results">
      <thead>
        <tr>
          <th>Rank</th><th>Category</th>
          <th>Strength Response to change</th>
          <th>Resistant Behaviour risk</th>
          <th>Reframing Strength narrative</th>
        </tr>
      </thead>
      <tbody>${changeRows}</tbody>
    </table>
  </section>

  <section class="page">
    ${header(10, 'Change Alignment Guide')}
    <div class="info">${copy.howToReadChangeAlignment}</div>
  </section>
</body></html>`;
}

export function teamFilename(
  companyName: string,
  dateStr: string,
): string {
  const safe = companyName.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 40);
  return `SSM_Team_${safe}_${dateStr}_v1.pdf`;
}
