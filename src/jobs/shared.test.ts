import { describe, it, expect } from 'vitest';
import { isFinalAttempt, participantName, recipientEmail, reportDate } from './shared.js';

describe('participantName', () => {
  it('prefers the demographics name over the assessment name', () => {
    expect(participantName({ participant_name: 'Old Name' }, { name: 'Ada Lovelace' })).toEqual({
      first: 'Ada',
      last: 'Lovelace',
    });
  });

  it('treats runs of spaces or tabs between names as one separator', () => {
    expect(participantName({}, { name: '  Ada \t  Lovelace ' })).toEqual({ first: 'Ada', last: 'Lovelace' });
  });

  it('keeps multi-word surnames together', () => {
    expect(participantName({}, { name: 'Ludwig van Beethoven' })).toEqual({ first: 'Ludwig', last: 'van Beethoven' });
  });

  it('uses the first name as the last name for a single-word name, so file names never end in "undefined"', () => {
    expect(participantName({}, { name: 'Prince' })).toEqual({ first: 'Prince', last: 'Prince' });
  });

  it('falls back to a generic name when none was given (the free flow makes the name optional)', () => {
    expect(participantName({ participant_name: '  ' }, {})).toEqual({ first: 'Participant', last: 'Name' });
  });
});

describe('reportDate', () => {
  it('uses the submission day', () => {
    expect(reportDate({ submitted_at: '2026-11-19T23:59:00Z' })).toBe('2026-11-19');
  });

  it('falls back to today when the assessment has no submission time', () => {
    expect(reportDate({})).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

describe('recipientEmail', () => {
  it('prefers the demographics email, then the assessment email, else empty', () => {
    expect(recipientEmail({ participant_email: 'a@x.com' }, { email: 'b@x.com' })).toBe('b@x.com');
    expect(recipientEmail({ participant_email: 'a@x.com' }, {})).toBe('a@x.com');
    expect(recipientEmail({}, {})).toBe('');
  });
});

describe('isFinalAttempt', () => {
  it('is true only for the last of the configured attempts (attemptsMade counts finished ones)', () => {
    expect(isFinalAttempt(0, 3)).toBe(false);
    expect(isFinalAttempt(1, 3)).toBe(false);
    expect(isFinalAttempt(2, 3)).toBe(true);
  });

  it('treats a job with no attempts setting as a single attempt', () => {
    expect(isFinalAttempt(0, undefined)).toBe(true);
  });
});
