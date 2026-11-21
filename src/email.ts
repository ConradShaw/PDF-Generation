import { env } from './env.js';
import { escapeHtml } from './html.js';
import { individualCopy } from './copy/index.js';
import { teamCopy } from './copy/index.js';
import { careerCopy, profileCopy, roleFitCopy } from './copy/index.js';

export async function sendEmailViaResend(opts: {
  to: string;
  subject: string;
  html: string;
  attachments?: Array<{ filename: string; content: string }>;
}): Promise<void> {
  if (!env.resendApiKey) {
    console.warn(`[email] RESEND_API_KEY not set – skipping email to ${opts.to}`);
    return;
  }

  const payload: Record<string, unknown> = {
    from: 'support@shawsight.com',
    to: [opts.to],
    subject: opts.subject,
    html: opts.html,
  };
  if (opts.attachments?.length) {
    payload.attachments = opts.attachments;
  }

  const resp = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.resendApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Resend failed: ${resp.status} ${text}`);
  }
  const data = (await resp.json()) as { id?: string };
  console.log(`[email] sent to ${opts.to} (resend id: ${data.id})`);
}

export async function sendIndividualReportEmail(
  email: string,
  name: string,
  assessmentId: string,
  pdfBytes: Buffer,
  pdfFilename: string,
): Promise<void> {
  const e = individualCopy.email;
  const bullets = e.bullets.map((b) => `<li>${b}</li>`).join('');
  const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Your SSM Assessment Report</title></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
  <h1 style="color: #007697;">${e.heading}</h1>
  <p>Hello${name ? ' ' + escapeHtml(name) : ''},</p>
  <p>${e.intro}</p>
  <div style="background-color: #e6f7f8; border-left: 4px solid #007697; padding: 15px; margin: 20px 0;">
    <h2 style="margin-top: 0; color: #007697;">What’s Inside Your Report:</h2>
    <ul style="margin-bottom: 0;">${bullets}</ul>
  </div>
  <p><strong>Assessment ID:</strong> ${assessmentId}</p>
  <p style="margin-top: 30px;">
    If you have any questions about your report, please reach out to
    <a href="mailto:support@shawsight.com" style="color: #007697;">support@shawsight.com</a>
  </p>
  <p style="margin-top: 30px;">Best regards,<br><strong>Conrad Shaw</strong><br>Shaw Strengths Matrix®</p>
</div>
</body>
</html>`;

  await sendEmailViaResend({
    to: email,
    subject: e.subject,
    html,
    attachments: [
      {
        filename: pdfFilename,
        content: pdfBytes.toString('base64'),
      },
    ],
  });
}

function reportEmailHtml(opts: {
  heading: string;
  name: string;
  intro: string;
  bullets: string[];
  extraHtml?: string;
}): string {
  const bullets = opts.bullets.map((b) => `<li>${b}</li>`).join('');
  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>${opts.heading}</title></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
  <h1 style="color: #007697;">${opts.heading}</h1>
  <p>Hello${opts.name ? ' ' + escapeHtml(opts.name) : ''},</p>
  <p>${opts.intro}</p>
  <div style="background-color: #e6f7f8; border-left: 4px solid #007697; padding: 15px; margin: 20px 0;">
    <h2 style="margin-top: 0; color: #007697;">What’s Inside:</h2>
    <ul style="margin-bottom: 0;">${bullets}</ul>
  </div>
  ${opts.extraHtml ?? ''}
  <p style="margin-top: 30px;">
    If you have any questions, please reach out to
    <a href="mailto:support@shawsight.com" style="color: #007697;">support@shawsight.com</a>
  </p>
  <p style="margin-top: 30px;">Best regards,<br><strong>Conrad Shaw</strong><br>Shaw Strengths Matrix®</p>
</div>
</body>
</html>`;
}

/** Free Strength Profile email: PDF attached plus the link to unlock Career Matches. */
export async function sendProfileReportEmail(
  email: string,
  name: string,
  pdfBytes: Buffer,
  pdfFilename: string,
  unlockUrl: string,
): Promise<void> {
  const e = profileCopy.email;
  const extraHtml = `<div style="margin: 24px 0; text-align: center;">
    <p>${e.unlockIntro}</p>
    <a href="${escapeHtml(unlockUrl)}" style="display:inline-block;background:#007697;color:#fff;padding:12px 22px;text-decoration:none;font-weight:bold;border-radius:4px;">${e.unlockButton}</a>
  </div>`;
  await sendEmailViaResend({
    to: email,
    subject: e.subject,
    html: reportEmailHtml({ heading: e.heading, name, intro: e.intro, bullets: e.bullets, extraHtml }),
    attachments: [{ filename: pdfFilename, content: pdfBytes.toString('base64') }],
  });
}

/** Paid Career Matches email: full report attached. */
export async function sendCareerMatchEmail(
  email: string,
  name: string,
  pdfBytes: Buffer,
  pdfFilename: string,
): Promise<void> {
  const e = careerCopy.email;
  await sendEmailViaResend({
    to: email,
    subject: e.subject,
    html: reportEmailHtml({ heading: e.heading, name, intro: e.intro, bullets: e.bullets }),
    attachments: [{ filename: pdfFilename, content: pdfBytes.toString('base64') }],
  });
}

/** Role Fit report email to the buyer, once every invited candidate has finished. */
export async function sendRoleFitReportEmail(
  email: string,
  organisation: string,
  pdfBytes: Buffer,
  pdfFilename: string,
): Promise<void> {
  const e = roleFitCopy.email;
  await sendEmailViaResend({
    to: email,
    subject: e.subject,
    html: reportEmailHtml({ heading: e.heading, name: organisation, intro: e.intro, bullets: e.bullets }),
    attachments: [{ filename: pdfFilename, content: pdfBytes.toString('base64') }],
  });
}

export async function sendTeamReportEmail(
  email: string,
  companyName: string,
  companyCode: string,
  teamSize: number,
  pdfBytes: Buffer,
  pdfFilename: string,
): Promise<void> {
  const generated = new Date().toLocaleString('en-AU', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Team Assessment Report</title></head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<div style="max-width: 600px; margin: 0 auto; padding: 20px;">
  <h1 style="color: #007697;">${teamCopy.email.heading}</h1>
  <p>Hello,</p>
  <p>The team assessment report for <strong>${companyName}</strong> has been generated and is now available.</p>
  <div style="background-color: #e6f7f8; border-left: 4px solid #007697; padding: 15px; margin: 20px 0;">
    <p style="margin: 0;"><strong>Report Details:</strong></p>
    <ul style="margin: 10px 0; padding-left: 20px;">
      <li>Company: ${companyName}</li>
      <li>Company Code: ${companyCode}</li>
      <li>Team Members: ${teamSize}</li>
      <li>Generated: ${generated}</li>
    </ul>
  </div>
  <p>If you have any questions about your team report, please contact us at
    <a href="mailto:support@shawsight.com" style="color: #007697;">support@shawsight.com</a>
  </p>
  <p>Best regards,<br><strong>Shaw Strengths Matrix® Team</strong></p>
</div>
</body>
</html>`;

  await sendEmailViaResend({
    to: email,
    subject: `${teamCopy.email.subjectPrefix}${companyName}`,
    html,
    attachments: [
      {
        filename: pdfFilename,
        content: pdfBytes.toString('base64'),
      },
    ],
  });
}
