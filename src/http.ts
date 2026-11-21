import type { Context } from 'hono';

/** The JSON body of a request as an object; anything unparseable or not an object (null, array, text) becomes {}. */
export async function readJsonBody(c: Context): Promise<Record<string, unknown>> {
  const body: unknown = await c.req.json().catch(() => null);
  return body && typeof body === 'object' && !Array.isArray(body) ? (body as Record<string, unknown>) : {};
}
