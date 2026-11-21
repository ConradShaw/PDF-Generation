import type { Trait } from '../scoring/traits.js';

/** Holland codes: Realistic, Investigative, Artistic, Social, Enterprising, Conventional. */
export type Riasec = { R: number; I: number; A: number; S: number; E: number; C: number };

/** An O*NET occupation expressed in SSM terms (see adapter.ts). */
export type Occupation = {
  code: string;
  title: string;
  /** When O*NET last updated this occupation's Work Styles, as O*NET writes it (e.g. "12/2025"). */
  profileDate?: string;
  /** O*NET Occupational Interests, 1-7 per area. */
  riasec: Riasec;
  /** How much the occupation demands each SSM strength, 0-100. */
  requirements: Record<Trait, number>;
};

/** What a template may show: rank and stars only, never the underlying percentage. */
export type RatedOccupation = { code: string; title: string; rank: number; stars: number };
