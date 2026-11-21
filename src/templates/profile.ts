import { individualCopy, profileCopy } from '../copy/index.js';
import { escapeHtml } from '../html.js';
import { fileSafe } from '../filenames.js';
import {
  BASE_CSS,
  buildRows,
  formatLongDate,
  logoDataUri,
  rowsHtml,
} from './individual.js';

const PROFILE_CSS = `
  .archetype { border: 2px solid #4d93d9; background: #eef6fd; padding: 14px; margin: 14px 0; font-size: 12px; line-height: 1.5; }
  .archetype h3 { margin: 0 0 6px; font-size: 15px; }
  .locked { border: 2px dashed #ba9354; padding: 14px; margin-top: 16px; font-size: 11.5px; line-height: 1.5; }
  .locked h3 { margin: 0 0 6px; font-size: 15px; }
  .cta { display: inline-block; margin-top: 8px; background: #007697; color: #fff; padding: 8px 14px; text-decoration: none; font-weight: bold; }
`;

/**
 * Free report: 12 ranked strengths, Work Archetype for the #1 strength only, and the unlock link.
 * `orderedTraits[0]` is the #1 strength; on a rank-1 tie the tie-breaker has already put the
 * alphabetically-first tied strength there.
 */
export function renderProfileHtml(opts: {
  first: string;
  last: string;
  dateStr: string;
  orderedTraits: string[];
  ranks: Record<string, number>;
  /** Archetype(s) of the #1 strength: several when strengths tie for first. */
  archetypes: Array<{ trait: string; text: string }>;
  unlockUrl: string;
}): string {
  const copy = profileCopy;
  const name = `${opts.first} ${opts.last}`;
  const logo = logoDataUri();
  const rows = buildRows(opts.orderedTraits, opts.ranks);
  const table = rowsHtml(
    rows,
    (row) =>
      `<td>${individualCopy.descriptions[row.trait as keyof typeof individualCopy.descriptions] ?? ''}</td>`,
  );
  const bullets = copy.lockedBullets.map((b) => `<li>${b}</li>`).join('');

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>${BASE_CSS}${PROFILE_CSS}</style></head>
<body>
  <section class="page">
    <h1>${copy.reportTitle}</h1>
    <div class="gold-bar"></div>
    <div class="box">
      <div><b>Prepared for: ${escapeHtml(name)}</b></div>
      <div><b>Date: ${escapeHtml(formatLongDate(opts.dateStr))}</b></div>
    </div>

    <div class="archetype">
      <h3>${copy.archetypeHeading}</h3>
      <p>${(opts.archetypes.length > 1 ? copy.archetypeIntroTied : copy.archetypeIntro).replace(
        '{trait}',
        opts.archetypes.map((a) => `<b>${escapeHtml(a.trait)}</b>`).join(' and '),
      )}</p>
      ${opts.archetypes.map((a) => `<p>${opts.archetypes.length > 1 ? `<b>${escapeHtml(a.trait)}:</b> ` : ''}${escapeHtml(a.text)}</p>`).join('')}
    </div>

    <h2>${copy.strengthsHeading}</h2>
    <div style="height:8px"></div>
    <table class="results">
      <thead><tr><th>Rank</th><th>Category</th><th>SSM Strength®</th><th>Description</th></tr></thead>
      <tbody>${table}</tbody>
    </table>
    <div class="info">${copy.strengthsNote}</div>

    <div class="locked">
      <h3>${copy.lockedHeading}</h3>
      <p>${copy.lockedIntro}</p>
      <ul>${bullets}</ul>
      <a class="cta" href="${escapeHtml(opts.unlockUrl)}">${copy.unlockButton}</a>
      <div style="margin-top:6px; font-size:9px; word-break:break-all;">${escapeHtml(opts.unlockUrl)}</div>
    </div>

    <div class="footer-row" style="margin-top:20px">
      ${logo ? `<img src="${logo}" alt="ShawSight"/>` : ''}
      <div>ShawSight Pty Ltd &nbsp;|&nbsp; ABN: 38688414557</div>
    </div>
  </section>
</body></html>`;
}

export function profileFilename(first: string, last: string, dateStr: string): string {
  return `SSM_Strength_Profile_${fileSafe(first)}_${fileSafe(last)}_${dateStr}.pdf`;
}
