import { describe, it, expect, vi, beforeEach } from 'vitest';
import { env } from '../src/env.js';
import { sendCareerMatchEmail, sendProfileReportEmail } from '../src/email.js';

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

const sentHtml = () => JSON.parse(fetchMock.mock.calls[0][1].body).html as string;

beforeEach(() => {
  fetchMock.mockReset().mockResolvedValue({ ok: true, json: async () => ({ id: 'r1' }) });
  vi.spyOn(console, 'log').mockImplementation(() => {});
  env.resendApiKey = 'test-key';
});

const evil = '<a href="https://evil.test">Verify your account</a>';

describe('report emails', () => {
  it('escape the recipient name: the free flow lets anyone request an email to any address with any name', async () => {
    await sendProfileReportEmail('victim@example.com', evil, Buffer.from('x'), 'f.pdf', 'https://app.test/u');
    expect(sentHtml()).not.toContain('<a href="https://evil.test">');
    expect(sentHtml()).toContain('&lt;a href=');

    fetchMock.mockClear();
    await sendCareerMatchEmail('victim@example.com', evil, Buffer.from('x'), 'f.pdf');
    expect(sentHtml()).not.toContain('<a href="https://evil.test">');
  });

  it('put the unlock link on the free profile email and attach the PDF', async () => {
    await sendProfileReportEmail('a@example.com', 'Ada', Buffer.from('pdf'), 'f.pdf', 'https://app.test/career-match/unlock?assessment=1');
    const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(payload.to).toEqual(['a@example.com']);
    expect(payload.html).toContain('https://app.test/career-match/unlock?assessment=1');
    expect(payload.attachments[0]).toMatchObject({ filename: 'f.pdf', content: Buffer.from('pdf').toString('base64') });
  });

  it('do not mention an unlock link in the paid report email', async () => {
    await sendCareerMatchEmail('a@example.com', 'Ada', Buffer.from('pdf'), 'f.pdf');
    expect(sentHtml()).not.toContain('Unlock');
  });

  it('surface a Resend failure instead of swallowing it, so the job can retry', async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 422, text: async () => 'bad' });
    await expect(sendCareerMatchEmail('a@example.com', 'Ada', Buffer.from('x'), 'f.pdf')).rejects.toThrow(/Resend failed/);
  });
});

describe('paid individual report email', () => {
  it('escapes the recipient name too (same injection route as the free flow)', async () => {
    const { sendIndividualReportEmail } = await import('../src/email.js');
    await sendIndividualReportEmail('victim@example.com', evil, 'assess-1', Buffer.from('x'), 'f.pdf');
    expect(sentHtml()).not.toContain('<a href="https://evil.test">');
  });
});
