"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

interface CatalogOption {
  id: number;
  name: string;
}

interface RecommendedSubcontractor {
  sub_id: number;
  name: string;
  primary_trade_id: number;
  certification_id: number;
  headcount?: number;
  years_in_business?: number;
}

interface TradeRecommendation {
  trade_id: number;
  trade_name: string;
  claim_probability: number | null;
  predicted_claim_risk?: number | null;
  baseline_claim_probability?: number | null;
  baseline_predicted_claim_risk?: number | null;
  risk_reduction_vs_baseline?: number | null;
  baseline_subcontractor?: { sub_id: number; name: string } | null;
  selection_strategy?: string;
  subcontractor: RecommendedSubcontractor | null;
}

interface ApiResponse {
  zip_code_id: number;
  project_type_id: number;
  requested_trade_ids: number[];
  project_team: TradeRecommendation[];
  error?: string;
}

interface ModelComponentStatus {
  loaded: boolean;
  checkpoint_path: string | null;
  error: string | null;
  latest_eval?: {
    metrics?: {
      policy?: {
        baseline_mean_risk?: number;
        ranker_mean_risk?: number;
        risk_reduction_pct?: number;
      };
    };
  } | null;
  best_metrics?: {
    precision?: number;
    recall?: number;
    accuracy?: number;
    loss?: number;
    policy?: {
      baseline_mean_risk?: number;
      ranker_mean_risk?: number;
      risk_reduction_pct?: number;
      baseline_low_risk_rate?: number;
      ranker_low_risk_rate?: number;
      baseline_ranking_target_rate?: number;
      ranker_ranking_target_rate?: number;
    };
  } | null;
}

interface ModelStatusResponse {
  recall_model: ModelComponentStatus;
  ranker_model: ModelComponentStatus;
  subcontractor_source_path?: string | null;
}

interface ProjectIntake {
  zipCodeId: number;
  projectTypeId: number;
  tradeIds: number[];
}

const BACKEND_BASE_URL = "http://localhost:8000";

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

const normalize = (value: string): string => value.toLowerCase().trim();

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const hasKeyword = (text: string, keyword: string): boolean => {
  const cleaned = keyword.trim();
  if (!cleaned) {
    return false;
  }
  const pattern = `\\b${escapeRegex(cleaned).replace(/\s+/g, "\\s+")}\\b`;
  return new RegExp(pattern, "i").test(text);
};

const tradeKeywordHints = (tradeName: string): string[] => {
  const key = normalize(tradeName);
  const tokens = key
    .split(/[^a-z0-9]+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3);
  const hints = new Set(tokens);

  if (key.includes("electrical")) {
    ["wiring", "wire", "electric", "power"].forEach((word) => hints.add(word));
  }
  if (key.includes("plumbing")) {
    ["pipe", "piping", "plumber", "water", "drain"].forEach((word) => hints.add(word));
  }
  if (key.includes("concrete")) {
    ["foundation", "slab", "rebar", "footing"].forEach((word) => hints.add(word));
  }
  if (key.includes("roof")) {
    ["roofing", "shingle", "waterproofing"].forEach((word) => hints.add(word));
  }
  if (key.includes("hvac")) {
    ["mechanical", "air", "duct", "ventilation"].forEach((word) => hints.add(word));
  }
  if (key.includes("framing")) {
    ["frame", "wood", "stud"].forEach((word) => hints.add(word));
  }
  if (key.includes("excavation")) {
    ["grading", "earthwork", "trenching", "sitework"].forEach((word) => hints.add(word));
  }
  if (key.includes("masonry")) {
    ["block", "brick", "stone"].forEach((word) => hints.add(word));
  }
  if (key.includes("drywall")) {
    ["gypsum", "sheetrock"].forEach((word) => hints.add(word));
  }
  if (key.includes("painting")) {
    ["paint", "coating", "finishes"].forEach((word) => hints.add(word));
  }

  return Array.from(hints);
};

const projectTypeKeywordHints = (projectTypeName: string): string[] => {
  const key = normalize(projectTypeName);
  const tokens = key
    .split(/[^a-z0-9]+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 3);
  const hints = new Set(tokens);

  if (key.includes("commercial")) {
    ["retail", "office", "mixed use", "hospitality"].forEach((word) => hints.add(word));
  }
  if (key.includes("residential")) {
    ["apartment", "multifamily", "single family", "home"].forEach((word) => hints.add(word));
  }
  if (key.includes("industrial")) {
    ["warehouse", "manufacturing", "plant", "distribution"].forEach((word) => hints.add(word));
  }
  if (key.includes("infrastructure") || key.includes("civil")) {
    ["bridge", "road", "highway", "utility", "transit"].forEach((word) => hints.add(word));
  }

  return Array.from(hints);
};

const parseScopeToIds = (
  scopeText: string,
  zipOptions: CatalogOption[],
  projectTypeOptions: CatalogOption[],
  tradeOptions: CatalogOption[],
): ProjectIntake => {
  const normalizedText = normalize(scopeText);
  if (!normalizedText) {
    return { zipCodeId: 0, projectTypeId: 0, tradeIds: [] };
  }

  let zipCodeId = 0;
  const zipMatches = normalizedText.match(/\b\d{5}\b/g) ?? [];
  for (const zip of zipMatches) {
    const option = zipOptions.find((candidate) => candidate.name.includes(zip));
    if (option) {
      zipCodeId = option.id;
      break;
    }
  }
  if (zipCodeId === 0) {
    for (const option of zipOptions) {
      const cityMatch = option.name.match(/\(([^)]+)\)/);
      const city = cityMatch?.[1]?.split(",")[0]?.trim().toLowerCase();
      if (city && hasKeyword(normalizedText, city)) {
        zipCodeId = option.id;
        break;
      }
    }
  }

  let projectTypeId = 0;
  let bestProjectTypeScore = 0;
  for (const option of projectTypeOptions) {
    const hits = projectTypeKeywordHints(option.name).filter((hint) => hasKeyword(normalizedText, hint));
    if (hits.length > bestProjectTypeScore) {
      bestProjectTypeScore = hits.length;
      projectTypeId = option.id;
    }
  }

  const tradeIds = tradeOptions
    .filter((option) => tradeKeywordHints(option.name).some((hint) => hasKeyword(normalizedText, hint)))
    .map((option) => option.id);

  return {
    zipCodeId,
    projectTypeId,
    tradeIds,
  };
};

const formatPercent = (value?: number | null, digits = 1): string =>
  typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "N/A";

const formatSignedPctPoints = (value?: number | null, digits = 1): string =>
  typeof value === "number"
    ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)} pts`
    : "N/A";

const riskTierLabel = (risk?: number | null): string => {
  if (typeof risk !== "number") {
    return "Unavailable";
  }
  if (risk <= 0.24) {
    return "Low";
  }
  if (risk <= 0.34) {
    return "Moderate";
  }
  return "Elevated";
};

export default function Home() {
  const [zipOptions, setZipOptions] = useState<CatalogOption[]>([]);
  const [projectTypeOptions, setProjectTypeOptions] = useState<CatalogOption[]>([]);
  const [tradeOptions, setTradeOptions] = useState<CatalogOption[]>([]);

  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const [scopeText, setScopeText] = useState("");
  const debouncedScopeText = useDebounce(scopeText, 300);

  const [intake, setIntake] = useState<ProjectIntake>({
    zipCodeId: 0,
    projectTypeId: 0,
    tradeIds: [],
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [modelStatusError, setModelStatusError] = useState<string | null>(null);

  const fetchCatalog = useCallback(async (path: string): Promise<CatalogOption[]> => {
    const res = await fetch(`${BACKEND_BASE_URL}${path}`);
    if (!res.ok) {
      throw new Error(`Failed to load catalog from ${path}`);
    }
    const rows = (await res.json()) as CatalogOption[];
    return rows.map((row) => ({ id: Number(row.id), name: row.name }));
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadCatalogs = async () => {
      setCatalogLoading(true);
      setCatalogError(null);

      try {
        const [zipRows, projectTypeRows, tradeRows] = await Promise.all([
          fetchCatalog("/api/zip-codes"),
          fetchCatalog("/api/project-types"),
          fetchCatalog("/api/trades"),
        ]);

        if (cancelled) {
          return;
        }

        setZipOptions(zipRows);
        setProjectTypeOptions(projectTypeRows);
        setTradeOptions(tradeRows);
      } catch (err) {
        if (!cancelled) {
          setCatalogError(err instanceof Error ? err.message : "Failed to load catalogs");
        }
      } finally {
        if (!cancelled) {
          setCatalogLoading(false);
        }
      }
    };

    loadCatalogs();
    return () => {
      cancelled = true;
    };
  }, [fetchCatalog]);

  const fetchModelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_BASE_URL}/api/model-status`);
      if (!res.ok) {
        throw new Error("Failed to load model status");
      }
      const json = (await res.json()) as ModelStatusResponse;
      setModelStatus(json);
      setModelStatusError(null);
    } catch (err) {
      setModelStatusError(err instanceof Error ? err.message : "Failed to load model status");
      setModelStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchModelStatus();
  }, [fetchModelStatus]);

  useEffect(() => {
    if (catalogLoading || catalogError) {
      return;
    }
    const parsed = parseScopeToIds(debouncedScopeText, zipOptions, projectTypeOptions, tradeOptions);
    setIntake(parsed);
    setError(null);
  }, [catalogLoading, catalogError, debouncedScopeText, zipOptions, projectTypeOptions, tradeOptions]);

  useEffect(() => {
    setError(null);
    setHasSubmitted(false);
  }, [intake.zipCodeId, intake.projectTypeId, intake.tradeIds]);

  const fetchRecommendations = useCallback(async (payload: ProjectIntake) => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_BASE_URL}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zip_code_id: payload.zipCodeId,
          project_type_id: payload.projectTypeId,
          trade_ids: payload.tradeIds,
        }),
      });
      const json: ApiResponse = await res.json();

      if (json.error) {
        setError(json.error);
        setData(null);
      } else {
        setData(json);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch recommendations");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const zipLabelById = useMemo(() => new Map(zipOptions.map((option) => [option.id, option.name])), [zipOptions]);
  const projectTypeLabelById = useMemo(
    () => new Map(projectTypeOptions.map((option) => [option.id, option.name])),
    [projectTypeOptions],
  );
  const tradeLabelById = useMemo(
    () => new Map(tradeOptions.map((option) => [option.id, option.name])),
    [tradeOptions],
  );

  const rankerPolicy =
    modelStatus?.ranker_model?.latest_eval?.metrics?.policy ??
    modelStatus?.ranker_model?.best_metrics?.policy;
  const recallMetrics = modelStatus?.recall_model?.best_metrics;
  const projectSummary = useMemo(() => {
    const team = data?.project_team ?? [];
    const staffed = team.filter((slot) => Boolean(slot.subcontractor));
    const predictedRisks = staffed
      .map((slot) => slot.predicted_claim_risk)
      .filter((value): value is number => typeof value === "number");
    const avgPredictedRisk =
      predictedRisks.length > 0
        ? predictedRisks.reduce((sum, value) => sum + value, 0) / predictedRisks.length
        : null;

    const riskDeltas = staffed
      .map((slot) => slot.risk_reduction_vs_baseline)
      .filter((value): value is number => typeof value === "number");
    const totalRiskReduction =
      riskDeltas.length > 0 ? riskDeltas.reduce((sum, value) => sum + value, 0) : null;
    const improvedTrades = riskDeltas.filter((value) => value > 0).length;

    return {
      requestedTrades: team.length,
      staffedTrades: staffed.length,
      avgPredictedRisk,
      totalRiskReduction,
      improvedTrades,
    };
  }, [data]);

  const updateTradeSelection = (selectedValues: string[]) => {
    const nextTradeIds = selectedValues
      .map((value) => parseInt(value, 10))
      .filter((value) => !Number.isNaN(value));
    setIntake((prev) => ({ ...prev, tradeIds: nextTradeIds }));
  };

  const onFindProjectTeam = () => {
    setHasSubmitted(true);
    if (intake.zipCodeId <= 0) {
      setError("Missing zip code. Add one in scope text or use Advanced Override.");
      setData(null);
      return;
    }
    if (intake.projectTypeId <= 0) {
      setError("Missing project type. Add one in scope text or use Advanced Override.");
      setData(null);
      return;
    }
    if (intake.tradeIds.length === 0) {
      setError("Missing required trades. Add them in scope text or use Advanced Override.");
      setData(null);
      return;
    }
    fetchRecommendations(intake);
  };

  return (
    <div className="dashboard">
      <h1 className="dashboard-header">Insurance Project Team Recommendations</h1>
      <p className="dashboard-subheader">Recommended trade partners ranked for lower expected claim risk</p>

      <section className="impact-section">
        <h2 className="project-vector-title">Model Impact (Held-out Validation)</h2>
        {modelStatusError && <p className="impact-meta">Model status unavailable: {modelStatusError}</p>}
        {!modelStatusError && !modelStatus && <p className="impact-meta">Loading model impact…</p>}
        {modelStatus && (
          <div className="impact-grid">
            <article className="impact-card">
              <h3 className="impact-card-title">Candidate Recall</h3>
              <p className="impact-meta">
                Status: {modelStatus.recall_model.loaded ? "Loaded checkpoint" : "Baseline weights"}
              </p>
              <p className="impact-stat">
                Shortlist hit rate:{" "}
                {typeof recallMetrics?.precision === "number"
                  ? `${(recallMetrics.precision * 100).toFixed(1)}%`
                  : "N/A"}
              </p>
              <p className="impact-stat">
                Candidate coverage:{" "}
                {typeof recallMetrics?.recall === "number"
                  ? `${(recallMetrics.recall * 100).toFixed(1)}%`
                  : "N/A"}
              </p>
            </article>

            <article className="impact-card">
              <h3 className="impact-card-title">Risk-Optimized Selection</h3>
              <p className="impact-meta">
                Status: {modelStatus.ranker_model.loaded ? "Loaded checkpoint" : "Recall-only fallback"}
              </p>
              <p className="impact-stat">
                Expected risk reduction vs recall-only:{" "}
                {typeof rankerPolicy?.risk_reduction_pct === "number"
                  ? `${rankerPolicy.risk_reduction_pct.toFixed(2)}%`
                  : "N/A"}
              </p>
              <div className="impact-bar-wrap">
                <span className="impact-bar-label">
                  Recall-only average risk:{" "}
                  {typeof rankerPolicy?.baseline_mean_risk === "number"
                    ? rankerPolicy.baseline_mean_risk.toFixed(3)
                    : "N/A"}
                </span>
                <div className="impact-bar-bg">
                  <div
                    className="impact-bar impact-bar-baseline"
                    style={{
                      width:
                        typeof rankerPolicy?.baseline_mean_risk === "number"
                          ? `${Math.max(2, Math.min(100, rankerPolicy.baseline_mean_risk * 100))}%`
                          : "0%",
                    }}
                  />
                </div>
              </div>
              <div className="impact-bar-wrap">
                <span className="impact-bar-label">
                  Ranker average risk:{" "}
                  {typeof rankerPolicy?.ranker_mean_risk === "number"
                    ? rankerPolicy.ranker_mean_risk.toFixed(3)
                    : "N/A"}
                </span>
                <div className="impact-bar-bg">
                  <div
                    className="impact-bar impact-bar-ranker"
                    style={{
                      width:
                        typeof rankerPolicy?.ranker_mean_risk === "number"
                          ? `${Math.max(2, Math.min(100, rankerPolicy.ranker_mean_risk * 100))}%`
                          : "0%",
                    }}
                  />
                </div>
              </div>
            </article>
          </div>
        )}
      </section>

      <section className="project-vector-section">
        <h2 className="project-vector-title">Underwriting Intake</h2>
        <div className="project-vector-field project-vector-field-wide">
          <label className="project-vector-label" htmlFor="scopeText">
            Paste Project Scope or Brief
          </label>
          <textarea
            id="scopeText"
            value={scopeText}
            onChange={(e) => setScopeText(e.target.value)}
            className="project-textarea project-scope-textarea"
            placeholder="Example: Commercial tenant improvement in 90001 with electrical rewiring, concrete slab repair, and HVAC replacement."
            rows={7}
            disabled={catalogLoading || Boolean(catalogError)}
          />
        </div>

        <h3 className="project-vector-title project-subtitle">Extracted Parameters</h3>
        <div className="project-chip-row">
          {intake.zipCodeId > 0 && (
            <span className="project-chip">📍 {zipLabelById.get(intake.zipCodeId) ?? `Zip ${intake.zipCodeId}`}</span>
          )}
          {intake.projectTypeId > 0 && (
            <span className="project-chip">
              🏢 {projectTypeLabelById.get(intake.projectTypeId) ?? `Project Type ${intake.projectTypeId}`}
            </span>
          )}
          {intake.tradeIds.map((tradeId) => (
            <span key={tradeId} className="project-chip">
              🛠️ {tradeLabelById.get(tradeId) ?? `Trade ${tradeId}`}
            </span>
          ))}
          {intake.zipCodeId <= 0 && intake.projectTypeId <= 0 && intake.tradeIds.length === 0 && (
            <span className="project-chip project-chip-muted">No parameters extracted yet.</span>
          )}
        </div>

        <button
          type="button"
          className="project-advanced-toggle"
          onClick={() => setShowAdvanced((prev) => !prev)}
          disabled={catalogLoading || Boolean(catalogError)}
        >
          {showAdvanced ? "Hide Advanced Override" : "Edit / Advanced Override"}
        </button>

        {showAdvanced && (
          <div className="project-advanced-panel">
            <div className="project-vector-grid">
              <div className="project-vector-field">
                <label className="project-vector-label" htmlFor="zipCodeId">
                  Zip code
                </label>
                <select
                  id="zipCodeId"
                  value={String(intake.zipCodeId)}
                  onChange={(e) =>
                    setIntake((prev) => ({ ...prev, zipCodeId: parseInt(e.target.value, 10) }))
                  }
                  className="project-select"
                  disabled={catalogLoading || zipOptions.length === 0}
                >
                  <option value="0">Select zip code</option>
                  {zipOptions.map((zip) => (
                    <option key={zip.id} value={zip.id}>
                      {zip.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="project-vector-field">
                <label className="project-vector-label" htmlFor="projectTypeId">
                  Project type
                </label>
                <select
                  id="projectTypeId"
                  value={String(intake.projectTypeId)}
                  onChange={(e) =>
                    setIntake((prev) => ({ ...prev, projectTypeId: parseInt(e.target.value, 10) }))
                  }
                  className="project-select"
                  disabled={catalogLoading || projectTypeOptions.length === 0}
                >
                  <option value="0">Select project type</option>
                  {projectTypeOptions.map((projectType) => (
                    <option key={projectType.id} value={projectType.id}>
                      {projectType.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="project-vector-field project-vector-field-wide">
                <label className="project-vector-label" htmlFor="tradeIds">
                  Required trades
                </label>
                <select
                  id="tradeIds"
                  multiple
                  value={intake.tradeIds.map(String)}
                  onChange={(e) =>
                    updateTradeSelection(Array.from(e.target.selectedOptions, (opt) => opt.value))
                  }
                  className="project-select project-select-multi"
                  disabled={catalogLoading || tradeOptions.length === 0}
                >
                  {tradeOptions.map((trade) => (
                    <option key={trade.id} value={trade.id}>
                      {trade.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}

        <div className="project-action-row">
          <button
            type="button"
            className="project-submit-button"
            onClick={onFindProjectTeam}
            disabled={catalogLoading || Boolean(catalogError) || loading}
          >
            Find Project Team
          </button>
        </div>
      </section>

      {catalogLoading && <div className="dashboard-loading">Loading intake catalogs…</div>}

      {catalogError && !catalogLoading && <div className="dashboard-error">{catalogError}</div>}

      {!catalogLoading && !catalogError && loading && (
        <div className="dashboard-loading">Building project team…</div>
      )}

      {error && !loading && !catalogLoading && !catalogError && (
        <div className="dashboard-error">
          {error}. Make sure the backend is running: <code>npm run dev:backend</code> (or{" "}
          <code>cd backend && uvicorn main:app --reload</code>)
        </div>
      )}

      {!loading && !error && hasSubmitted && !catalogLoading && !catalogError && data?.project_team && (
        <>
          <section className="team-summary-grid">
            <article className="team-summary-card">
              <p className="team-summary-label">Trades requested</p>
              <p className="team-summary-value">{projectSummary.requestedTrades}</p>
            </article>
            <article className="team-summary-card">
              <p className="team-summary-label">Trades staffed</p>
              <p className="team-summary-value">{projectSummary.staffedTrades}</p>
            </article>
            <article className="team-summary-card">
              <p className="team-summary-label">Avg projected claim risk</p>
              <p className="team-summary-value">{formatPercent(projectSummary.avgPredictedRisk)}</p>
            </article>
            <article className="team-summary-card">
              <p className="team-summary-label">Total risk reduction vs recall baseline</p>
              <p className="team-summary-value">{formatSignedPctPoints(projectSummary.totalRiskReduction)}</p>
              {projectSummary.totalRiskReduction !== null && (
                <p className="team-summary-note">{projectSummary.improvedTrades} trade slots improved</p>
              )}
            </article>
          </section>

          <div className="subcontractor-feed">
            {data.project_team.map((teamSlot) => (
              <article
                key={`trade-${teamSlot.trade_id}`}
                className={`subcontractor-card ${teamSlot.subcontractor ? "subcontractor-card-tier1" : ""}`}
              >
                <span className="tier1-badge">{teamSlot.trade_name}</span>

                {teamSlot.subcontractor ? (
                  <>
                    <h2 className="subcontractor-name">{teamSlot.subcontractor.name}</h2>
                    <p className="subcontractor-claim-prob">
                      Projected claim risk: {formatPercent(teamSlot.predicted_claim_risk)}
                    </p>
                    <p className="subcontractor-risk-tier">
                      Risk tier: {riskTierLabel(teamSlot.predicted_claim_risk)}
                    </p>
                    {typeof teamSlot.risk_reduction_vs_baseline === "number" && (
                      <p className="subcontractor-impact">
                        Reduction vs recall baseline: {formatSignedPctPoints(teamSlot.risk_reduction_vs_baseline)}
                      </p>
                    )}
                    {(typeof teamSlot.subcontractor.headcount === "number" ||
                      typeof teamSlot.subcontractor.years_in_business === "number") && (
                      <p className="subcontractor-stat">
                        {typeof teamSlot.subcontractor.headcount === "number"
                          ? `Crew size: ${teamSlot.subcontractor.headcount}`
                          : ""}
                        {typeof teamSlot.subcontractor.headcount === "number" &&
                        typeof teamSlot.subcontractor.years_in_business === "number"
                          ? " | "
                          : ""}
                        {typeof teamSlot.subcontractor.years_in_business === "number"
                          ? `Years in business: ${teamSlot.subcontractor.years_in_business}`
                          : ""}
                      </p>
                    )}
                  </>
                ) : (
                  <p className="subcontractor-stat">No subcontractor available for this trade.</p>
                )}
              </article>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
