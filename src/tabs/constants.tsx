const EXPECTED_MARKET_LAMBDA = 0.6;

const parseEnvFloat = (value: string | undefined, fallback: number): number => {
  if (value === undefined || value === null || value === "") return fallback;
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

export const EDGE_MIN = parseEnvFloat(import.meta.env.VITE_EDGE_MIN as string | undefined, 1.8);
export const VALUE_MIN = parseEnvFloat(
  import.meta.env.VITE_VALUE_MIN as string | undefined,
  Number(Math.max(0.8, EDGE_MIN * Math.max(0, 1 - EXPECTED_MARKET_LAMBDA)).toFixed(2))
);

export const BETS_EDGE_MIN = parseEnvFloat(
  import.meta.env.VITE_BETS_EDGE_MIN as string | undefined,
  1.8
);
export const BETS_VALUE_MIN = parseEnvFloat(
  import.meta.env.VITE_BETS_VALUE_MIN as string | undefined,
  Number(Math.max(0.8, BETS_EDGE_MIN * Math.max(0, 1 - EXPECTED_MARKET_LAMBDA)).toFixed(2))
);

export const CONFIDENCE_MIN = parseEnvFloat(
  import.meta.env.VITE_CONFIDENCE_MIN as string | undefined,
  0.55
);
export const HIGH_CONFIDENCE_MIN = parseEnvFloat(
  import.meta.env.VITE_HIGH_CONFIDENCE_MIN as string | undefined,
  0.7
);
export const HIGH_CONF_VALUE_MIN = parseEnvFloat(
  import.meta.env.VITE_HIGH_CONF_VALUE_MIN as string | undefined,
  0.5
);
export const LARGE_SPREAD_ABS_MIN = parseEnvFloat(
  import.meta.env.VITE_LARGE_SPREAD_ABS_MIN as string | undefined,
  14.0
);
export const LARGE_SPREAD_CONF_MIN = parseEnvFloat(
  import.meta.env.VITE_LARGE_SPREAD_CONF_MIN as string | undefined,
  0.7
);

export const MODEL_EXPECTED_LAMBDA = EXPECTED_MARKET_LAMBDA;
