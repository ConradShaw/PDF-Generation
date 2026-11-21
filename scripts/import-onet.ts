/**
 * Build data/onet-occupations.json from the public O*NET database CSVs.
 * Usage: tsx scripts/import-onet.ts <dir-with-work_styles.csv-and-career_interest_types.csv> <db-version-label>
 * Download: https://www.onetcenter.org/database.html (CSV format).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { buildOccupations, parseCsv } from '../src/careers/onetImport.js';
import mapping from '../data/ssm-mapping.json' with { type: 'json' };

const [dir, version] = process.argv.slice(2);
if (!dir || !version) {
  console.error('usage: tsx scripts/import-onet.ts <csv-dir> <db-version-label, e.g. "31.0">');
  process.exit(1);
}

const read = (f: string) => parseCsv(readFileSync(join(dir, f), 'utf8'));
const workStyles = read('work_styles.csv');
const interests = read('career_interest_types.csv');
const occupations = buildOccupations(workStyles, interests, mapping.strengthToWorkStyle);

const out = {
  source: `O*NET ${version} Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). Used under the CC BY 4.0 license. O*NET is a trademark of USDOL/ETA.`,
  occupations,
};
writeFileSync(new URL('../data/onet-occupations.json', import.meta.url), JSON.stringify(out));
console.log(`Wrote ${occupations.length} occupations (of ${new Set(workStyles.map((r) => r['O*NET-SOC Code'])).size} with work styles, ${new Set(interests.map((r) => r['O*NET-SOC Code'])).size} with interests)`);
