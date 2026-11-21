/** First/last name for a report: demographics name, else the assessment's participant name, else a generic fallback. */
export function participantName(
  assessment: Record<string, unknown>,
  demographics: Record<string, unknown>,
): { first: string; last: string } {
  const fullName = String(demographics.name ?? assessment.participant_name ?? '');
  if (!fullName.trim()) return { first: 'Participant', last: 'Name' };
  const parts = fullName.trim().split(/\s+/);
  return { first: parts[0], last: parts.length > 1 ? parts.slice(1).join(' ') : parts[0] };
}

/** YYYY-MM-DD the assessment was submitted, or today. */
export function reportDate(assessment: Record<string, unknown>): string {
  return (
    String(assessment.submitted_at ?? '').slice(0, 10) ||
    new Date().toISOString().slice(0, 10)
  );
}

export function recipientEmail(
  assessment: Record<string, unknown>,
  demographics: Record<string, unknown>,
): string {
  return String(demographics.email ?? assessment.participant_email ?? '');
}

/**
 * True when the attempt now running is the last one BullMQ will make. `attemptsMade` counts
 * finished attempts, so the running one is attemptsMade + 1; a job with no setting runs once.
 */
export function isFinalAttempt(attemptsMade: number, attempts: number | undefined): boolean {
  return attemptsMade + 1 >= (attempts ?? 1);
}
