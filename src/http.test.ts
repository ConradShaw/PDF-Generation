import { describe, it, expect } from 'vitest';
import type { Context } from 'hono';
import { readJsonBody } from './http.js';

const ctx = (json: () => Promise<unknown>) => ({ req: { json } }) as unknown as Context;

describe('readJsonBody', () => {
  it('returns the parsed object', async () => {
    expect(await readJsonBody(ctx(async () => ({ assessment_id: 'a1' })))).toEqual({ assessment_id: 'a1' });
  });

  it('turns an unparseable body into {} so the route answers "id is required" (400), not a 500', async () => {
    expect(await readJsonBody(ctx(async () => { throw new SyntaxError('Unexpected token'); }))).toEqual({});
  });

  it('turns null, arrays and plain values into {}, because the routes read fields from the body', async () => {
    for (const odd of [null, [1, 2], 'text', 42, true]) {
      expect(await readJsonBody(ctx(async () => odd))).toEqual({});
    }
  });
});
