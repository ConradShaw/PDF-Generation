import { describe, it, expect } from 'vitest';
import { escapeHtml } from './html.js';

describe('escapeHtml', () => {
  it('escapes all four characters that can break out of HTML text or a quoted attribute', () => {
    expect(escapeHtml('<a href="x">&</a>')).toBe('&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;');
  });

  it('escapes ampersands first, so existing entities are shown literally, not decoded or double-encoded wrongly', () => {
    expect(escapeHtml('&lt;')).toBe('&amp;lt;');
  });

  it('leaves ordinary text, accents and apostrophes untouched', () => {
    expect(escapeHtml("Zoë's plan")).toBe("Zoë's plan");
  });
});
