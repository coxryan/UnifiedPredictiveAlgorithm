const EXPECTED_MARKET_LAMBDA = 0.6;

const parseEnvFloat = (value: string | undefined, fallback: number): number => {
  if (value === undefined || value === null || value === "") return fallback;
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

export const EDGE_MIN = parseEnvFloat(import.meta.env.VITE_EDGE_MIN as string | undefined, 2.0);
export const VALUE_MIN = parseEnvFloat(
  import.meta.env.VITE_VALUE_MIN as string | undefined,
  Number((EDGE_MIN * Math.max(0, 1 - EXPECTED_MARKET_LAMBDA)).toFixed(2))
);

export const BETS_EDGE_MIN = parseEnvFloat(
  import.meta.env.VITE_BETS_EDGE_MIN as string | undefined,
  1.5
);
export const BETS_VALUE_MIN = parseEnvFloat(
  import.meta.env.VITE_BETS_VALUE_MIN as string | undefined,
  Number((BETS_EDGE_MIN * Math.max(0, 1 - EXPECTED_MARKET_LAMBDA)).toFixed(2))
);

export const MODEL_EXPECTED_LAMBDA = EXPECTED_MARKET_LAMBDA;
