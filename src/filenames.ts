/**
 * A name part made safe for a storage key and a file name: accents folded to ASCII
 * ("José" -> "Jose"), anything else outside [A-Za-z0-9._-] replaced with "_". Supabase Storage
 * rejects non-ASCII keys, so a raw "Zoë" would make the upload (and the job) fail.
 * Falls back to "Report" if nothing usable remains (e.g. a name written only in Chinese).
 */
export function fileSafe(part: string): string {
  const folded = part
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return folded || 'Report';
}
