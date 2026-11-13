import {
  TRAITS,
  categoryForRankNumber,
  type Category,
} from './traits.js';

export type IndividualResult = {
  ordered_traits?: string[];
  ranks: Record<string, number>;
};

export type DistributionData = Record<string, Record<Category, number>>;

function median(nums: number[]): number {
  if (nums.length === 0) return 12;
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

/**
 * Port of calculate_team_rankings.
 */
export function calculateTeamRankings(individualResults: IndividualResult[]): {
  orderedTraits: string[];
  finalRanks: Record<string, number>;
  distributionData: DistributionData;
} {
  const rankSums: Record<string, number> = {};
  const rankCounts: Record<string, number> = {};
  const individualRanksByTrait: Record<string, number[]> = {};
  for (const trait of TRAITS) {
    rankSums[trait] = 0;
    rankCounts[trait] = 0;
    individualRanksByTrait[trait] = [];
  }

  const normalized = individualResults.map((r) => ({
    ranks: r.ranks ?? {},
  }));

  for (const result of normalized) {
    for (const [trait, rank] of Object.entries(result.ranks)) {
      if (!(trait in rankSums)) continue;
      const numericRank = Number(rank);
      rankSums[trait] += numericRank;
      rankCounts[trait] += 1;
      individualRanksByTrait[trait].push(numericRank);
    }
  }

  const avgRanks: Record<string, number> = {};
  const medianRanks: Record<string, number> = {};
  for (const trait of TRAITS) {
    avgRanks[trait] =
      rankCounts[trait] > 0 ? rankSums[trait] / rankCounts[trait] : 12;
    medianRanks[trait] = median(individualRanksByTrait[trait]);
  }

  const orderedTraits = [...TRAITS].sort((a, b) => {
    const avgDiff = round6(avgRanks[a]) - round6(avgRanks[b]);
    if (avgDiff !== 0) return avgDiff;
    const medDiff = round6(medianRanks[a]) - round6(medianRanks[b]);
    if (medDiff !== 0) return medDiff;
    return a.localeCompare(b);
  });

  const finalRanks: Record<string, number> = {};
  let nominalPosition = 1;
  let i = 0;
  while (i < orderedTraits.length) {
    const key = tieKey(avgRanks, medianRanks, orderedTraits[i]);
    const group = [orderedTraits[i]];
    while (
      i + group.length < orderedTraits.length &&
      tieKey(avgRanks, medianRanks, orderedTraits[i + group.length]) === key
    ) {
      group.push(orderedTraits[i + group.length]);
    }
    const positions = Array.from(
      { length: group.length },
      (_, k) => nominalPosition + k,
    );
    const avgPosition = mean(positions);
    for (const trait of group) {
      finalRanks[trait] = avgPosition;
    }
    nominalPosition += group.length;
    i += group.length;
  }

  const totalCount = individualResults.length;
  const distributionData: DistributionData = {};
  for (const trait of TRAITS) {
    const categoryCounts: Record<Category, number> = {
      Signature: 0,
      Supporting: 0,
      Stretch: 0,
      Situational: 0,
    };
    for (const result of normalized) {
      const rank = Number(result.ranks[trait] ?? 12);
      categoryCounts[categoryForRankNumber(rank)] += 1;
    }
    distributionData[trait] = {
      Signature: totalCount > 0 ? categoryCounts.Signature / totalCount : 0,
      Supporting: totalCount > 0 ? categoryCounts.Supporting / totalCount : 0,
      Stretch: totalCount > 0 ? categoryCounts.Stretch / totalCount : 0,
      Situational: totalCount > 0 ? categoryCounts.Situational / totalCount : 0,
    };
  }

  return { orderedTraits, finalRanks, distributionData };
}

function round6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

function tieKey(
  avgRanks: Record<string, number>,
  medianRanks: Record<string, number>,
  trait: string,
): string {
  return `${round6(avgRanks[trait])}|${round6(medianRanks[trait])}`;
}
