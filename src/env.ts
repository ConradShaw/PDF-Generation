import { loadLocalEnv } from './localEnv.js';

// Tests must not pick up a developer's real .env file.
if (!process.env.VITEST) loadLocalEnv();

export function getEnv(name: string, fallback = ''): string {
  return process.env[name] ?? fallback;
}

export const env = {
  port: Number(getEnv('PORT', '8080')),
  redisUrl: getEnv('REDIS_URL', 'redis://127.0.0.1:6379'),
  supabaseUrl: getEnv('SUPABASE_URL'),
  supabaseServiceRoleKey: getEnv('SUPABASE_SERVICE_ROLE_KEY'),
  resendApiKey: getEnv('RESEND_API_KEY'),
  disableReportEmails: ['1', 'true', 'yes', 'on'].includes(
    getEnv('DISABLE_REPORT_EMAILS', '').toLowerCase(),
  ),
  reportsBucket: getEnv('REPORTS_BUCKET', 'reports'),
  /** Public site URL, used for the "unlock Career Matches" link in the free report email. */
  appUrl: getEnv('APP_URL', 'http://localhost:5173').replace(/\/+$/, ''),
  isProduction: getEnv('NODE_ENV') === 'production',
  /** Optional basic-auth for Bull Board (`/admin/queues`). Both must be set to enable. */
  bullBoardUser: getEnv('BULL_BOARD_USER'),
  bullBoardPassword: getEnv('BULL_BOARD_PASSWORD'),
};
