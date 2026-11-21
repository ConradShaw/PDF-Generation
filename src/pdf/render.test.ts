import { describe, it, expect, vi, beforeEach } from 'vitest';

const page = { setContent: vi.fn(), pdf: vi.fn(async () => new Uint8Array([37, 80, 68, 70])), close: vi.fn() };
const browser = { newPage: vi.fn(async () => page), close: vi.fn() };
const launch = vi.fn(async () => browser);
vi.mock('playwright', () => ({ chromium: { launch: (...a: unknown[]) => (launch as (...x: unknown[]) => unknown)(...a) } }));

beforeEach(() => {
  vi.clearAllMocks();
  vi.resetModules(); // the browser is cached per process, so each test starts a fresh module
});

describe('htmlToPdf browser launch', () => {
  it('does not let Playwright kill Chromium on SIGTERM/SIGINT/SIGHUP: a deploy would fail the report being rendered, instead of letting the worker finish it first', async () => {
    const { htmlToPdf } = await import('./render.js');
    await htmlToPdf('<p>x</p>');
    expect(launch).toHaveBeenCalledWith(expect.objectContaining({ handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false }));
  });

  it('launches one browser for many reports and closes each page it opens', async () => {
    const { htmlToPdf } = await import('./render.js');
    await htmlToPdf('<p>1</p>');
    await htmlToPdf('<p>2</p>');
    expect(launch).toHaveBeenCalledTimes(1);
    expect(page.close).toHaveBeenCalledTimes(2);
  });

  it('closes the page even when rendering fails, so a failed job cannot leak pages', async () => {
    page.setContent.mockRejectedValueOnce(new Error('boom'));
    const { htmlToPdf } = await import('./render.js');
    await expect(htmlToPdf('<p>x</p>')).rejects.toThrow('boom');
    expect(page.close).toHaveBeenCalledTimes(1);
  });

  it('closeBrowser closes the browser once and lets a later report launch a new one', async () => {
    const { htmlToPdf, closeBrowser } = await import('./render.js');
    await htmlToPdf('<p>x</p>');
    await closeBrowser();
    await closeBrowser(); // second call is a no-op
    expect(browser.close).toHaveBeenCalledTimes(1);
    await htmlToPdf('<p>y</p>');
    expect(launch).toHaveBeenCalledTimes(2);
  });
});
