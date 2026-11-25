import { roleFitCopy, individualCopy } from '../copy/index.js';
import { fileSafe } from '../filenames.js';
import { escapeHtml } from '../html.js';
import type { RatedCandidate, RoleProfileRow } from '../rolefit/engine.js';
import { colorForCategory, formatRank } from '../scoring/traits.js';
import { starsHtml } from './career.js';
import { BASE_CSS, formatLongDate, logoDataUri } from './individual.js';
import { strengthsMatrixHtml } from './matrix.js';

const ROLE_FIT_CSS = `
  table.fit { width: 100%; border-collapse: collapse; font-size: 10px; }
  table.fit th, table.fit td { border: 1px solid #000; padding: 5px 6px; vertical-align: middle; }
  table.fit th { background: #f3f3f3; }
  table.fit tr { page-break-inside: avoid; }
  .occ { margin: 6px 0 10px; font-size: 11px; }
  .legend p { margin: 2px 0; }
`;

/** Paragraphs of copy: lines starting with "* " or "a) " keep their own line within one paragraph. */
const paragraphs = (items: readonly string[]) =>
  items.map((p) => `<p>${escapeHtml(p).replace(/\n/g, '<br/>')}</p>`).join('');

export function roleFitFilename(organisation: string, dateStr: string): string {
  return `SSM_Role_Fit_${fileSafe(organisation)}_${dateStr}.pdf`;
}

export function renderRoleFitHtml(opts: {
  organisation: string;
  team: string;
  numMembers: number;
  dateStr: string;
  occupation: { title: string; profileDate?: string };
  profile: RoleProfileRow[];
  candidates: RatedCandidate[];
}): string {
  const c = roleFitCopy;
  const logo = logoDataUri();
  const org = escapeHtml(opts.organisation);
  const team = escapeHtml(opts.team);
  const header = (page: number, subtitle: string) => `
    <div class="hdr"><div>Shaw Strengths Matrix® Assessment</div><div>${team} | Page ${page}</div></div>
    <h2>${subtitle}</h2><div style="height:8px"></div>`;

  const profileRows = opts.profile
    .map((row, i) => {
      const spacer = i > 0 && opts.profile[i - 1].category !== row.category ? '<tr class="spacer"><td colspan="4"></td></tr>' : '';
      const definition = individualCopy.descriptions[row.trait as keyof typeof individualCopy.descriptions] ?? '';
      return `${spacer}<tr>
        <td class="center cat" style="background:${colorForCategory(row.category)}">${formatRank(row.rank)}</td>
        <td class="center cat" style="background:${colorForCategory(row.category)}">${row.category}</td>
        <td class="center"><b>${row.trait}</b></td><td>${definition}</td></tr>`;
    })
    .join('');

  const candidateRows = opts.candidates
    .map(
      (cand) => `<tr>
        <td class="center">${cand.rank}</td>
        <td>${escapeHtml(cand.name)}</td>
        <td class="center">${starsHtml(cand.stars)}</td>
        <td class="center">${cand.score.toFixed(1)}%</td>
        <td class="center">${cand.fitLabel}</td>
        <td>${escapeHtml(cand.aligned.join(', '))}</td>
        <td>${escapeHtml(cand.misaligned.join(', '))}</td></tr>`,
    )
    .join('');

  const profileDate = opts.occupation.profileDate
    ? ` &nbsp;&nbsp;&nbsp; ONET profile date: ${escapeHtml(opts.occupation.profileDate)}`
    : '';
  const heads = c.tableHeads;

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>${BASE_CSS}${ROLE_FIT_CSS}</style></head>
<body>
  <section class="page">
    <h1>${c.title}</h1>
    <div class="gold-bar"></div>
    <div class="box">
      <div><b>${c.preparedFor}</b></div>
      <div><b>Organisation: ${org}</b></div>
      <div><b>Team: ${team}</b></div>
      <div><b>Number of Team members: ${opts.numMembers}</b></div>
      <div><b>Date: ${escapeHtml(formatLongDate(opts.dateStr))}</b></div>
    </div>
    <div class="footer-row">
      ${logo ? `<img src="${logo}" alt="ShawSight"/>` : ''}
      <div>ShawSight Pty Ltd &nbsp;|&nbsp; ABN: 38688414557</div>
    </div>
    <div class="box" style="margin-top:12px">${paragraphs(c.legal)}</div>
  </section>

  <section class="page">
    ${header(2, c.overviewHeading)}
    <div class="box"><p><b>${c.aboutHeading}</b></p>${paragraphs(c.about)}</div>
    <div class="box"><p><b>${c.contentsHeading}</b></p>${paragraphs(c.contents)}</div>
  </section>

  <section class="page">
    ${header(3, c.matrixHeading)}
    ${strengthsMatrixHtml()}
    <div class="info"><p><b>${c.howToRead}</b></p>${paragraphs(c.matrix)}</div>
  </section>

  <section class="page">
    ${header(4, c.targetHeading)}
    <div class="occ"><b>Occupation: ${escapeHtml(opts.occupation.title)}${profileDate}</b></div>
    <table class="results">
      <thead><tr><th>${heads.rank}</th><th>${heads.category}</th><th>${heads.strength}</th><th>${heads.definition}</th></tr></thead>
      <tbody>${profileRows}</tbody>
    </table>
    <div class="info"><p><b>${c.howToRead}</b></p>${paragraphs(c.target)}</div>
  </section>

  <section class="page">
    ${header(5, c.fitHeading)}
    <table class="fit">
      <thead><tr><th>${heads.rank}</th><th>${heads.candidate}</th><th>${heads.rating}</th><th>${heads.score}</th><th>${heads.fitCategory}</th><th>${heads.aligned}</th><th>${heads.misaligned}</th></tr></thead>
      <tbody>${candidateRows}</tbody>
    </table>
    <div class="info"><p><b>${c.howToRead}</b></p>${paragraphs(c.fit)}<div class="legend">${paragraphs(c.starLegend)}</div>${paragraphs([c.weightingNote])}</div>
  </section>
</body></html>`;
}
