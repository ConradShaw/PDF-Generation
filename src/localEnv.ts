import { existsSync } from 'node:fs';

/**
 * Local development: read settings from a .env file so `npm run dev` / `npm start` work without extra flags.
 * Real environment variables always win, a missing file is fine, and production (Docker/Railway) has no .env
 * file, so this does nothing there. Returns true when a file was loaded.
 */
export function loadLocalEnv(path = '.env'): boolean {
  if (!existsSync(path) || typeof process.loadEnvFile !== 'function') return false;
  process.loadEnvFile(path); // does not override variables that are already set
  return true;
}
