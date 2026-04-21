import { useState, useEffect, useMemo } from "react";
import { api } from "../api";
import type { JobSummary, JobDetail, PerRecord, InsightPattern } from "../api";
import { Card, DataTable, Empty, MultiCheck, SelectField, SectionHeader } from "../components/ui";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  AreaChart,
  Area,
} from "recharts";

const COLORS = [
  "#39B54A",
  "#BE1E2D",
  "#1a8fbf",
  "#e6a817",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#6366f1",
];

const COLORS_ALPHA = [
  "rgba(57,181,74,0.7)",
  "rgba(190,30,45,0.7)",
  "rgba(26,143,191,0.7)",
  "rgba(230,168,23,0.7)",
  "rgba(139,92,246,0.7)",
  "rgba(236,72,153,0.7)",
  "rgba(20,184,166,0.7)",
  "rgba(99,102,241,0.7)",
];

const GRID_STROKE = "rgba(0,0,0,0.06)";

/* Custom glassmorphism tooltip */
function GlassTooltip({ active, payload, label, formatter }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-label">{label}</p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="chart-tooltip-row">
          <span className="chart-tooltip-dot" style={{ background: p.color || p.fill }} />
          <span className="chart-tooltip-name">{p.name || p.dataKey}</span>
          <span className="chart-tooltip-val">
            {formatter ? formatter(p.value) : (typeof p.value === "number" ? p.value.toFixed(3) : p.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* SVG gradient definitions for charts */
function ChartGradients() {
  return (
    <defs>
      <linearGradient id="grad-green" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#39B54A" stopOpacity={0.9} />
        <stop offset="100%" stopColor="#39B54A" stopOpacity={0.35} />
      </linearGradient>
      <linearGradient id="grad-red" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#BE1E2D" stopOpacity={0.9} />
        <stop offset="100%" stopColor="#BE1E2D" stopOpacity={0.35} />
      </linearGradient>
      <linearGradient id="grad-blue" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#1a8fbf" stopOpacity={0.9} />
        <stop offset="100%" stopColor="#1a8fbf" stopOpacity={0.35} />
      </linearGradient>
      <linearGradient id="grad-gold" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#e6a817" stopOpacity={0.9} />
        <stop offset="100%" stopColor="#e6a817" stopOpacity={0.35} />
      </linearGradient>
      <linearGradient id="grad-purple" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.9} />
        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.35} />
      </linearGradient>
      <linearGradient id="grad-area-green" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#39B54A" stopOpacity={0.3} />
        <stop offset="100%" stopColor="#39B54A" stopOpacity={0.02} />
      </linearGradient>
      <linearGradient id="grad-area-blue" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#1a8fbf" stopOpacity={0.3} />
        <stop offset="100%" stopColor="#1a8fbf" stopOpacity={0.02} />
      </linearGradient>
      {COLORS.map((c, i) => (
        <linearGradient key={i} id={`grad-bar-${i}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={c} stopOpacity={0.9} />
          <stop offset="100%" stopColor={c} stopOpacity={0.4} />
        </linearGradient>
      ))}
    </defs>
  );
}

/* Severity badge */
function SeverityBadge({ severity }: { severity: InsightPattern["severity"] }) {
  return (
    <span className={`severity-badge severity-badge--${severity}`}>
      {severity === "high" ? "🔴 High" : severity === "medium" ? "🟡 Medium" : "🟢 Low"}
    </span>
  );
}

/* Insight Agent full tab */
function InsightAgentTab({ insights }: { insights: import("../api").InsightData | null }) {
  if (!insights) {
    return (
      <Card>
        <Empty message="Insight Agent not available. Complete a job to generate the analysis." />
      </Card>
    );
  }

  const data = insights.insights;
  // error_patterns are at the top level of InsightData (from stats), not inside insights.insights
  const patterns: InsightPattern[] = insights.error_patterns ?? [];
  // operational_recommendations: prefer top-level, fallback to inside insights.insights
  const opRecs: string[] = insights.operational_recommendations ?? data?.operational_recommendations ?? [];
  const perTask: Record<string, string> = data?.per_task_analysis ?? {};

  return (
    <>
      {/* Summary + general recommendation */}
      {(data?.summary || data?.recommendation) && (
        <Card accent>
          <SectionHeader title="AI Summary" />
          {data.summary && (
            <div className="insight-ai-bubble insight-ai-summary mb-3">
              <span className="insight-ai-bubble-label">
                <span className="emoji-icon emoji-sparkle">✨</span> Summary
              </span>
              <p>{data.summary}</p>
            </div>
          )}
          {data.recommendation && (
            <div className="insight-ai-bubble insight-ai-reco">
              <span className="insight-ai-bubble-label">
                <span className="emoji-icon emoji-rocket">🚀</span> General Recommendation
              </span>
              <p>{data.recommendation}</p>
            </div>
          )}
        </Card>
      )}

      {/* Operational recommendations — BN_INSIGHT_AGENT */}
      {opRecs.length > 0 && (
        <Card>
          <SectionHeader title="Operative Recommendations" />
          <ul className="operative-recs-list">
            {opRecs.map((rec, i) => (
              <li key={i} className="operative-rec-item">
                <span className="operative-rec-icon">⚙️</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Error patterns with severity + test_ids — BN_INSIGHT_AGENT */}
      {patterns.length > 0 && (
        <Card>
          <SectionHeader title="Error Patterns" />
          <div className="pattern-list">
            {patterns.map((p, i) => (
              <div key={i} className={`pattern-card pattern-card--${p.severity}`}>
                <div className="pattern-card-header">
                  <SeverityBadge severity={p.severity} />
                  <span className="pattern-card-title">{p.description}</span>
                </div>
                {p.test_ids.length > 0 && (
                  <div className="pattern-test-ids">
                    <span className="pattern-test-ids-label">Involved test cases:</span>
                    {p.test_ids.map((tid) => (
                      <span key={tid} className="pattern-test-id-chip">{tid}</span>
                    ))}
                  </div>
                )}
                {p.recommendation && (
                  <div className="pattern-recommendation">
                    <span className="emoji-icon">💡</span> {p.recommendation}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Per-task analysis */}
      {Object.keys(perTask).length > 0 && (
        <Card>
          <SectionHeader title="Per-Task Analysis" />
          <div className="per-task-analysis">
            {Object.entries(perTask).map(([task, analysis]) => (
              <div key={task} className="per-task-item">
                <span className="per-task-label">{task}</span>
                <p className="per-task-text">{analysis}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Anomalies */}
      {data?.anomalies_analysis && (
        <Card>
          <SectionHeader title="Detected Anomalies" />
          <p className="text-muted">{data.anomalies_analysis}</p>
        </Card>
      )}

      {patterns.length === 0 && opRecs.length === 0 && Object.keys(perTask).length === 0 && !data?.summary && (
        <Card>
          <Empty message="No Insight Agent data available for this job." />
        </Card>
      )}
    </>
  );
}

export default function InsightsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [tab, setTab] = useState<"verdict" | "compare" | "perf">("verdict");

  // Filters for verdict tab
  const [filterTask, setFilterTask] = useState<string[]>([]);
  const [filterModel, setFilterModel] = useState<string[]>([]);

  /**
   * BN_ADVANCED_FRONTEND — Auto-polling every 4s (≤ 5s).
   * Refreshes the completed job list automatically so new results appear
   * without manual page reload.
   */
  const POLL_INTERVAL_MS = 4000;

  useEffect(() => {
    let cancelled = false;

    const fetchJobs = async () => {
      try {
        const d = await api.listJobs();
        if (cancelled) return;
        const completed = d.jobs.filter((j) => j.status === "completed");
        setJobs(completed);
        if (completed.length > 0 && !selectedId) setSelectedId(completed[0].job_id);
      } catch { /* ignore polling errors */ }
    };

    fetchJobs();
    const id = setInterval(fetchJobs, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;

    const fetchDetail = async () => {
      try {
        const d = await api.getJob(selectedId);
        if (!cancelled) setDetail(d);
      } catch { /* ignore */ }
    };

    fetchDetail();
    const id = setInterval(fetchDetail, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [selectedId]);

  const agg = detail?.aggregated;

  // --- Verdict data ---
  const verdictRows = useMemo(() => {
    if (!agg?.per_record) return [];
    let rows: PerRecord[] = agg.per_record;
    if (filterTask.length > 0) rows = rows.filter((r) => filterTask.includes(r.task));
    if (filterModel.length > 0) rows = rows.filter((r) => filterModel.includes(r.model_id));
    return rows;
  }, [agg, filterTask, filterModel]);

  const allTasks = useMemo(
    () => [...new Set(agg?.per_record?.map((r) => r.task) || [])].sort(),
    [agg]
  );
  const allModels = useMemo(
    () => [...new Set(agg?.per_record?.map((r) => r.model_id) || [])].sort(),
    [agg]
  );

  // --- Compare data ---
  const compareChartData = useMemo(() => {
    if (!agg?.per_task_model) return [];
    const taskMap: Record<string, Record<string, number>> = {};
    for (const v of Object.values(agg.per_task_model)) {
      if (!taskMap[v.task]) taskMap[v.task] = {};
      taskMap[v.task][v.model_id] = v.avg_score;
    }
    return Object.entries(taskMap).map(([task, models]) => ({
      task,
      ...models,
    }));
  }, [agg]);

  const modelScoreData = useMemo(() => {
    if (!agg?.per_model) return [];
    return Object.entries(agg.per_model)
      .map(([mid, info]) => ({
        model: mid.length > 25 ? mid.slice(0, 22) + "…" : mid,
        score: info.overall_score,
      }))
      .sort((a, b) => b.score - a.score);
  }, [agg]);

  const modelsInChart = useMemo(() => {
    if (!compareChartData.length) return [];
    const keys = new Set<string>();
    for (const d of compareChartData) {
      for (const k of Object.keys(d)) {
        if (k !== "task") keys.add(k);
      }
    }
    return [...keys];
  }, [compareChartData]);

  // --- Operative data ---
  const opRows = useMemo(() => {
    if (!agg?.operative_metrics) return [];
    return Object.values(agg.operative_metrics);
  }, [agg]);

  const latencyData = useMemo(() => {
    const map: Record<string, number[]> = {};
    for (const r of opRows) {
      if (!map[r.model]) map[r.model] = [];
      map[r.model].push(r.avg_latency_ms);
    }
    return Object.entries(map)
      .map(([model, vals]) => ({
        model: model.length > 25 ? model.slice(0, 22) + "…" : model,
        latency: vals.reduce((a, b) => a + b, 0) / vals.length,
      }))
      .sort((a, b) => b.latency - a.latency);
  }, [opRows]);

  const costData = useMemo(() => {
    const map: Record<string, number> = {};
    for (const r of opRows) {
      map[r.model] = (map[r.model] || 0) + r.total_cost;
    }
    return Object.entries(map)
      .map(([model, cost]) => ({
        model: model.length > 25 ? model.slice(0, 22) + "…" : model,
        cost,
      }))
      .sort((a, b) => b.cost - a.cost);
  }, [opRows]);

  const tokenData = useMemo(() => {
    const map: Record<string, { input: number; output: number }> = {};
    for (const r of opRows) {
      if (!map[r.model]) map[r.model] = { input: 0, output: 0 };
      map[r.model].input += r.input_tokens;
      map[r.model].output += r.output_tokens;
    }
    return Object.entries(map).map(([model, t]) => ({
      model: model.length > 25 ? model.slice(0, 22) + "…" : model,
      input_tokens: t.input,
      output_tokens: t.output,
    }));
  }, [opRows]);

  if (!jobs.length) {
    return (
      <div className="page">
        <div className="page-header">
          <h2 className="page-title">Insight & Confronto</h2>
        </div>
        <Empty message="No completed jobs available." />
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Insight & Compare</h2>
        <p className="page-subtitle">Detailed verdict analysis, model comparison and operative metrics</p>
      </div>

      <Card>
        <SelectField
          label="Select job"
          value={selectedId}
          onChange={setSelectedId}
          options={jobs.map((j) => ({
            value: j.job_id,
            label: `${j.job_id} — ${j.dataset_key}`,
          }))}
        />
      </Card>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${tab === "verdict" ? "active" : ""}`}
          onClick={() => setTab("verdict")}
        >
          📝 Verdict per Record
        </button>
        <button
          className={`tab ${tab === "compare" ? "active" : ""}`}
          onClick={() => setTab("compare")}
        >
          📊 Model Comparison
        </button>
        <button
          className={`tab ${tab === "perf" ? "active" : ""}`}
          onClick={() => setTab("perf")}
        >
          ⚡ Performance
        </button>

      </div>

      {/* Tab: Verdict */}
      {tab === "verdict" && (
        <Card>
          {agg?.per_record?.length ? (
            <>
              <div className="grid-2 mb-3">
                <MultiCheck
                  label="Filter by task"
                  options={allTasks.map((t) => ({ value: t, label: t }))}
                  selected={filterTask}
                  onChange={setFilterTask}
                />
                <MultiCheck
                  label="Filter by model"
                  options={allModels.map((m) => ({ value: m, label: m.length > 30 ? m.slice(0, 27) + "…" : m }))}
                  selected={filterModel}
                  onChange={setFilterModel}
                />
              </div>

              <DataTable
                maxHeight={500}
                columns={[
                  { key: "test_id", label: "Test ID" },
                  { key: "task", label: "Task" },
                  { key: "model_id", label: "Modello" },
                  { key: "score", label: "Score", align: "right" },
                  { key: "verdict", label: "Verdict", align: "center" },
                  { key: "output", label: "Output" },
                ]}
                rows={verdictRows.map((r) => ({
                  test_id: r.test_id,
                  task: r.task,
                  model_id:
                    r.model_id.length > 25
                      ? r.model_id.slice(0, 22) + "…"
                      : r.model_id,
                  score: r.score?.toFixed(3),
                  verdict: r.correct ? "✅" : "❌",
                  output: String(r.output_text || "").slice(0, 200),
                }))}
              />
              <p className="text-muted mt-2">
                {verdictRows.length} records displayed
              </p>
            </>
          ) : (
            <Empty message="Nessun record di valutazione disponibile." />
          )}
        </Card>
      )}

      {/* Tab: Compare */}
      {tab === "compare" && (
        <>
          {modelScoreData.length > 0 && (
            <Card>
              <SectionHeader title="Overall Score per Model" />
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={modelScoreData} barCategoryGap="20%">
                  <ChartGradients />
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip content={<GlassTooltip />} cursor={{ fill: "rgba(57,181,74,0.06)" }} />
                  <Bar
                    dataKey="score"
                    fill="url(#grad-green)"
                    radius={[8, 8, 0, 0]}
                    animationBegin={100}
                    animationDuration={1200}
                    animationEasing="ease-out"
                  >
                    {modelScoreData.map((_, i) => (
                      <Cell key={i} fill={`url(#grad-bar-${i % COLORS.length})`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {compareChartData.length > 0 && (
            <Card>
              <SectionHeader title="Score per Task (model comparison)" />
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={compareChartData} barCategoryGap="15%">
                  <ChartGradients />
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="task" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip content={<GlassTooltip />} cursor={{ fill: "rgba(57,181,74,0.06)" }} />
                  <Legend wrapperStyle={{ paddingTop: 12 }} />
                  {modelsInChart.map((m, i) => (
                    <Bar
                      key={m}
                      dataKey={m}
                      fill={`url(#grad-bar-${i % COLORS.length})`}
                      radius={[6, 6, 0, 0]}
                      animationBegin={100 + i * 150}
                      animationDuration={1000}
                      animationEasing="ease-out"
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Pivot table */}
          {compareChartData.length > 0 && (
            <Card>
              <SectionHeader title="Score per Model × Task" />
              <DataTable
                columns={[
                  { key: "model", label: "Modello" },
                  ...allTasks.map((t) => ({
                    key: t,
                    label: t,
                    align: "right" as const,
                  })),
                ]}
                rows={allModels.map((m) => {
                  const row: Record<string, unknown> = { model: m };
                  for (const t of allTasks) {
                    const key = Object.keys(agg!.per_task_model).find(
                      (k) =>
                        agg!.per_task_model[k].task === t &&
                        agg!.per_task_model[k].model_id === m
                    );
                    row[t] = key
                      ? agg!.per_task_model[key].avg_score.toFixed(3)
                      : "—";
                  }
                  return row;
                })}
              />
            </Card>
          )}
        </>
      )}

      {/* Tab: Performance */}
      {tab === "perf" && (
        <>
          {latencyData.length > 0 && (
            <Card>
              <SectionHeader title="⏱️ Average Latency per Model" />
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={latencyData} barCategoryGap="20%" layout="vertical">
                  <ChartGradients />
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} horizontal={false} />
                  <YAxis dataKey="model" type="category" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} width={150} />
                  <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip
                    content={<GlassTooltip formatter={(v: number) => `${v.toFixed(0)} ms`} />}
                    cursor={{ fill: "rgba(26,143,191,0.06)" }}
                  />
                  <Bar
                    dataKey="latency"
                    fill="url(#grad-blue)"
                    radius={[0, 8, 8, 0]}
                    animationBegin={100}
                    animationDuration={1200}
                    animationEasing="ease-out"
                  />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {costData.length > 0 && (
            <Card>
              <SectionHeader title="💰 Total Cost per Model" />
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={costData} barCategoryGap="20%">
                  <ChartGradients />
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip
                    content={<GlassTooltip formatter={(v: number) => `$${v.toFixed(4)}`} />}
                    cursor={{ fill: "rgba(230,168,23,0.06)" }}
                  />
                  <Bar
                    dataKey="cost"
                    fill="url(#grad-gold)"
                    radius={[8, 8, 0, 0]}
                    animationBegin={200}
                    animationDuration={1200}
                    animationEasing="ease-out"
                  >
                    {costData.map((_, i) => (
                      <Cell key={i} fill={`url(#grad-bar-${(i + 3) % COLORS.length})`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {tokenData.length > 0 && (
            <Card>
              <SectionHeader title="🔤 Token Usage per Model" />
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={tokenData}>
                  <ChartGradients />
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} vertical={false} />
                  <XAxis dataKey="model" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip content={<GlassTooltip />} cursor={{ stroke: "rgba(57,181,74,0.3)", strokeWidth: 1 }} />
                  <Legend wrapperStyle={{ paddingTop: 12 }} />
                  <Area
                    type="monotone"
                    dataKey="input_tokens"
                    stroke="#39B54A"
                    strokeWidth={2.5}
                    fill="url(#grad-area-green)"
                    name="Input Tokens"
                    animationBegin={100}
                    animationDuration={1400}
                    animationEasing="ease-out"
                    dot={{ r: 4, fill: "#39B54A", strokeWidth: 2, stroke: "#fff" }}
                    activeDot={{ r: 6, fill: "#39B54A", stroke: "#fff", strokeWidth: 2 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="output_tokens"
                    stroke="#1a8fbf"
                    strokeWidth={2.5}
                    fill="url(#grad-area-blue)"
                    name="Output Tokens"
                    animationBegin={300}
                    animationDuration={1400}
                    animationEasing="ease-out"
                    dot={{ r: 4, fill: "#1a8fbf", strokeWidth: 2, stroke: "#fff" }}
                    activeDot={{ r: 6, fill: "#1a8fbf", stroke: "#fff", strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          )}

          {opRows.length > 0 && (
            <Card>
              <SectionHeader title="📋 Operative Metrics Table" />
              <DataTable
                columns={[
                  { key: "model", label: "Modello" },
                  { key: "task", label: "Task" },
                  { key: "avg_latency_ms", label: "Latenza (ms)", align: "right" },
                  { key: "total_requests", label: "Richieste", align: "right" },
                  { key: "num_errors", label: "Errori", align: "right" },
                  { key: "input_tokens", label: "Input Tok", align: "right" },
                  { key: "output_tokens", label: "Output Tok", align: "right" },
                  { key: "total_cost", label: "Costo ($)", align: "right" },
                ]}
                rows={opRows.map((r) => ({
                  ...r,
                  avg_latency_ms: r.avg_latency_ms?.toFixed(0),
                  total_cost: r.total_cost?.toFixed(4),
                }))}
              />
            </Card>
          )}

          {opRows.length === 0 && (
            <Card>
              <Empty message="Operative metrics not available. Make sure the runner tracked latency and tokens." />
            </Card>
          )}
        </>
      )}

    </div>
  );
}
