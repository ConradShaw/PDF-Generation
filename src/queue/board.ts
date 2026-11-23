import { createBullBoard } from '@bull-board/api';
import { BullMQAdapter } from '@bull-board/api/bullMQAdapter';
import { HonoAdapter } from '@bull-board/hono';
import { serveStatic } from '@hono/node-server/serve-static';
import type { Hono } from 'hono';
import { basicAuth } from 'hono/basic-auth';
import { env } from '../env.js';
import {
  getCareerQueue,
  getIndividualQueue,
  getRoleFitQueue,
  getTeamQueue,
} from './queues.js';

/** URL path where the Bull Board UI is mounted. */
export const BULL_BOARD_BASE_PATH = '/admin/queues';

/**
 * Mounts Bull Board for all PDF queues.
 * When `BULL_BOARD_USER` + `BULL_BOARD_PASSWORD` are set, the UI is basic-auth protected.
 */
export function mountBullBoard(app: Hono): void {
  const serverAdapter = new HonoAdapter(serveStatic);
  serverAdapter.setBasePath(BULL_BOARD_BASE_PATH);

  createBullBoard({
    queues: [
      new BullMQAdapter(getIndividualQueue()),
      new BullMQAdapter(getTeamQueue()),
      new BullMQAdapter(getCareerQueue()),
      new BullMQAdapter(getRoleFitQueue()),
    ],
    serverAdapter,
  });

  if (env.bullBoardUser && env.bullBoardPassword) {
    const auth = basicAuth({
      username: env.bullBoardUser,
      password: env.bullBoardPassword,
    });
    app.use(BULL_BOARD_BASE_PATH, auth);
    app.use(`${BULL_BOARD_BASE_PATH}/*`, auth);
  }

  app.route(BULL_BOARD_BASE_PATH, serverAdapter.registerPlugin());
}
