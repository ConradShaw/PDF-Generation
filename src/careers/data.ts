import { env } from '../env.js';
import occupationsFile from '../../data/onet-occupations.json' with { type: 'json' };
import archetypesFile from '../../data/archetypes.json' with { type: 'json' };
import mappingFile from '../../data/ssm-mapping.json' with { type: 'json' };
import type { Occupation } from './types.js';

export const occupations = occupationsFile.occupations as Occupation[];

/** O*NET attribution required by the O*NET license; printed in the Career Match report. */
export const onetAttribution = occupationsFile.source;

/** The owner's Work Archetype text for a strength. */
export function archetypeFor(trait: string): string {
  const text = (archetypesFile.archetypes as Record<string, string>)[trait];
  if (!text) throw new Error(`No Work Archetype text for strength "${trait}"`);
  return text;
}

type OwnerData = {
  archetypes: { placeholder: boolean };
  mapping: { placeholder: boolean };
};

export function ownerDataProblems(data: OwnerData): string[] {
  const problems: string[] = [];
  if (data.archetypes.placeholder) {
    problems.push('data/archetypes.json still holds placeholder Work Archetype text');
  }
  if (data.mapping.placeholder) {
    problems.push('data/ssm-mapping.json is still an unconfirmed placeholder');
  }
  return problems;
}

/**
 * Production must never email placeholder content to a customer: fail the job loudly
 * (BullMQ retries and logs) instead. Development runs end to end with placeholders.
 */
export function assertOwnerDataReady(
  isProduction: boolean = env.isProduction,
  data: OwnerData = { archetypes: archetypesFile, mapping: mappingFile },
): void {
  const problems = ownerDataProblems(data);
  if (isProduction && problems.length) {
    throw new Error(`Refusing to send report with placeholder owner data: ${problems.join('; ')}`);
  }
}
