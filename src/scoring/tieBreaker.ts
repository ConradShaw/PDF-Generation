import { TRAITS, type SurveyResults } from './traits.js';

type RankGroup = string | string[];

function mean(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

/**
 * Faithful port of PDF-Generation/main.py TieBreaker.
 */
export class TieBreaker {
  traits: string[];
  results: SurveyResults;
  scores: Record<string, number>;

  constructor(traits: readonly string[], results: SurveyResults) {
    this.traits = [...traits];
    this.results = results;
    this.scores = {};
    for (const t of traits) {
      this.scores[t] = Object.values(results[t] ?? {}).reduce((a, b) => a + b, 0);
    }
  }

  rankWithAverageRanks(): { ordered: string[]; ranks: Record<string, number> } {
    const base = this.primaryOrder();
    const groups = this.resolveEqualScoreGroups(base);
    return this.assignAverageRanks(groups);
  }

  private primaryOrder(): string[] {
    return [...this.traits].sort((a, b) => {
      const scoreDiff = this.scores[b] - this.scores[a];
      if (scoreDiff !== 0) return scoreDiff;
      return a.localeCompare(b);
    });
  }

  private resolveEqualScoreGroups(baseOrder: string[]): RankGroup[] {
    const out: RankGroup[] = [];
    let i = 0;
    while (i < baseOrder.length) {
      const run = [baseOrder[i]];
      while (
        i + run.length < baseOrder.length &&
        this.scores[baseOrder[i]] === this.scores[baseOrder[i + run.length]]
      ) {
        run.push(baseOrder[i + run.length]);
      }

      if (run.length === 1) {
        out.push(run[0]);
      } else if (run.length === 2) {
        out.push(...this.headToHeadPair(run[0], run[1]));
      } else {
        out.push(...this.orderRunWithMinileague(run));
      }
      i += run.length;
    }
    return out;
  }

  private headToHeadPair(a: string, b: string): string[] {
    return (this.results[a]?.[b] ?? 0) === 1 ? [a, b] : [b, a];
  }

  private orderRunWithMinileague(run: string[]): RankGroup[] {
    const subs: Record<string, number> = {};
    for (const t of run) subs[t] = 0;
    for (const a of run) {
      for (const b of run) {
        if (a === b) continue;
        subs[a] += this.results[a]?.[b] ?? 0;
      }
    }

    const sortedBySub = [...run].sort((a, b) => {
      const d = subs[b] - subs[a];
      if (d !== 0) return d;
      return a.localeCompare(b);
    });

    const resolved: RankGroup[] = [];
    let i = 0;
    while (i < sortedBySub.length) {
      const block = [sortedBySub[i]];
      while (
        i + block.length < sortedBySub.length &&
        subs[sortedBySub[i]] === subs[sortedBySub[i + block.length]]
      ) {
        block.push(sortedBySub[i + block.length]);
      }

      if (block.length === 1) {
        resolved.push(block[0]);
      } else if (block.length === 2) {
        resolved.push(...this.headToHeadPair(block[0], block[1]));
      } else {
        resolved.push(...this.miniLeagueThenSosWithExtra(block));
      }
      i += block.length;
    }
    return resolved;
  }

  private miniLeagueThenSosWithExtra(block: string[]): RankGroup[] {
    const wins: Record<string, number> = {};
    for (const t of block) wins[t] = 0;
    for (const a of block) {
      for (const b of block) {
        if (a === b) continue;
        wins[a] += this.results[a]?.[b] ?? 0;
      }
    }

    const ordered = [...block].sort((a, b) => {
      const d = wins[b] - wins[a];
      if (d !== 0) return d;
      return a.localeCompare(b);
    });

    const out: RankGroup[] = [];
    let i = 0;
    while (i < ordered.length) {
      const band = [ordered[i]];
      while (
        i + band.length < ordered.length &&
        wins[ordered[i]] === wins[ordered[i + band.length]]
      ) {
        band.push(ordered[i + band.length]);
      }

      if (band.length === 1) {
        out.push(band[0]);
      } else if (band.length === 2) {
        out.push(...this.headToHeadPair(band[0], band[1]));
      } else {
        out.push(...this.sosThenExtraMinileagueOnTies(band));
      }
      i += band.length;
    }
    return out;
  }

  private sosThenExtraMinileagueOnTies(band: string[]): RankGroup[] {
    const sos: Record<string, number> = {};
    for (const t of band) {
      sos[t] = 0;
      for (const o of Object.keys(this.results[t] ?? {})) {
        if ((this.results[t]?.[o] ?? 0) === 1) {
          sos[t] += this.scores[o] ?? 0;
        }
      }
    }

    const ordered = [...band].sort((a, b) => {
      const d = sos[b] - sos[a];
      if (d !== 0) return d;
      return a.localeCompare(b);
    });

    const out: RankGroup[] = [];
    let i = 0;
    while (i < ordered.length) {
      const sub = [ordered[i]];
      while (
        i + sub.length < ordered.length &&
        sos[ordered[i]] === sos[ordered[i + sub.length]]
      ) {
        sub.push(ordered[i + sub.length]);
      }

      if (sub.length === 1) {
        out.push(sub[0]);
      } else if (sub.length === 2) {
        out.push(...this.headToHeadPair(sub[0], sub[1]));
      } else {
        out.push(...this.finalMinileagueOrTie(sub));
      }
      i += sub.length;
    }
    return out;
  }

  private finalMinileagueOrTie(subband: string[]): RankGroup[] {
    const wins: Record<string, number> = {};
    for (const t of subband) wins[t] = 0;
    for (const a of subband) {
      for (const b of subband) {
        if (a === b) continue;
        wins[a] += this.results[a]?.[b] ?? 0;
      }
    }

    const ordered = [...subband].sort((a, b) => {
      const d = wins[b] - wins[a];
      if (d !== 0) return d;
      return a.localeCompare(b);
    });

    const out: RankGroup[] = [];
    let i = 0;
    while (i < ordered.length) {
      const band = [ordered[i]];
      while (
        i + band.length < ordered.length &&
        wins[ordered[i]] === wins[ordered[i + band.length]]
      ) {
        band.push(ordered[i + band.length]);
      }

      if (band.length === 1) {
        out.push(band[0]);
      } else if (band.length === 2) {
        out.push(...this.headToHeadPair(band[0], band[1]));
      } else {
        out.push([...band].sort((a, b) => a.localeCompare(b)));
      }
      i += band.length;
    }
    return out;
  }

  private assignAverageRanks(groups: RankGroup[]): {
    ordered: string[];
    ranks: Record<string, number>;
  } {
    const ordered: string[] = [];
    const ranks: Record<string, number> = {};
    let pos = 1;

    for (const item of groups) {
      if (typeof item === 'string') {
        ordered.push(item);
        ranks[item] = pos;
        pos += 1;
      } else {
        const m = item.length;
        const avg = mean(Array.from({ length: m }, (_, k) => pos + k));
        for (const t of [...item].sort((a, b) => a.localeCompare(b))) {
          ordered.push(t);
          ranks[t] = avg;
        }
        pos += m;
      }
    }
    return { ordered, ranks };
  }
}

export function rankTraits(results: SurveyResults) {
  return new TieBreaker(TRAITS, results).rankWithAverageRanks();
}
