import { useState, useEffect } from "react";
import { api } from "../api";
import type { AgentInfo, ModelConfig, PromptOptResult } from "../api";
import {
  Button,
  Card,
  DataTable,
  Empty,
  Metric,
  SelectField,
  SectionHeader,
  Slider,
} from "../components/ui";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

const TOOLTIP_STYLE = {
  background: "rgba(255,255,255,0.92)",
  border: "1px solid rgba(57,181,74,0.2)",
  borderRadius: "8px",
  color: "#1a1a2e",
};

export default function PromptOptPage() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [selectedAgent, setSelectedAgent] = useState("");
  const [selectedModel, setSelectedModel] = useState("");

  const [maxIter, setMaxIter] = useState(3);
  const [numVariants, setNumVariants] = useState(3);
  const [beamWidth, setBeamWidth] = useState(2);

  /** BN_PROMPT_OPTIMIZATION — multi-objective weights (quality, cost, latency) */
  const [qualityWeight, setQualityWeight] = useState(0.7);
  const [costWeight, setCostWeight] = useState(0.2);
  const [latencyWeight, setLatencyWeight] = useState(0.1);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PromptOptResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [expandVariants, setExpandVariants] = useState(false);

  useEffect(() => {
    api.getDetected().then((d) => {
      if (d.detected) {
        setAgents(d.agents);
        if (d.agents.length > 0) setSelectedAgent(d.agents[0].agent_id);
      }
    }).catch(() => {});

    api.getModels().then((d) => {
      setModels(d.models);
      if (d.models.length > 0) setSelectedModel(d.models[0].model_id);
    }).catch(() => {});
  }, []);

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.promptOptimize({
        agent_id: selectedAgent,
        model_id: selectedModel,
        max_iterations: maxIter,
        num_variants: numVariants,
        beam_width: beamWidth,
        quality_weight: qualityWeight,
        cost_weight: costWeight,
        latency_weight: latencyWeight,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  if (!agents.length) {
    return (
      <div className="page">
        <div className="page-header">
          <h2 className="page-title">Prompt Optimization</h2>
        </div>
        <Empty message="Run Detect Tasks in Config Job first to discover available agents." />
      </div>
    );
  }

  const delta = result ? result.best_score - result.baseline_score : 0;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Prompt Optimization</h2>
        <p className="page-subtitle">Iterative prompt optimization via beam search</p>
      </div>

      {/* Agent & Model selection */}
      <Card>
        <SectionHeader title="Configuration" />
        <div className="grid-2">
          <SelectField
            label="Agent ID"
            value={selectedAgent}
            onChange={setSelectedAgent}
            options={agents.map((a) => ({
              value: a.agent_id,
              label: `${a.agent_id} (${a.task})`,
            }))}
          />
          <SelectField
            label="Candidate Model"
            value={selectedModel}
            onChange={setSelectedModel}
            options={models.map((m) => ({
              value: m.model_id,
              label: m.display_name || m.model_id,
            }))}
          />
        </div>
      </Card>

      {/* Parameters */}
      <Card>
        <SectionHeader title="Iteration Parameters" />
        <div className="grid-3">
          <Slider label="Max Iterations" value={maxIter} min={1} max={5} onChange={setMaxIter} />
          <Slider label="Variants per Iteration" value={numVariants} min={1} max={5} onChange={setNumVariants} />
          <Slider label="Beam Width" value={beamWidth} min={1} max={5} onChange={setBeamWidth} />
        </div>
      </Card>

      {/* Multi-objective weights — BN_PROMPT_OPTIMIZATION */}
      <Card>
        <SectionHeader title="Multi-Objective Weights" />
        <p className="text-muted mb-3" style={{ fontSize: "0.85rem" }}>
          Configure the relative importance of each objective in prompt optimization.
          Weights define the composite score: <code>score = quality×{qualityWeight.toFixed(1)} + cost×{costWeight.toFixed(1)} + latency×{latencyWeight.toFixed(1)}</code>
        </p>
        <div className="grid-3">
          <div className="weight-slider-wrap">
            <Slider
              label={`Quality (${(qualityWeight * 100).toFixed(0)}%)`}
              value={Math.round(qualityWeight * 10)}
              min={1} max={9}
              onChange={(v) => setQualityWeight(v / 10)}
            />
            <p className="weight-rationale">Prioritizes correctness and coherence of responses.</p>
          </div>
          <div className="weight-slider-wrap">
            <Slider
              label={`Cost (${(costWeight * 100).toFixed(0)}%)`}
              value={Math.round(costWeight * 10)}
              min={1} max={9}
              onChange={(v) => setCostWeight(v / 10)}
            />
            <p className="weight-rationale">Penalizes prompts that increase token consumption and per-request cost.</p>
          </div>
          <div className="weight-slider-wrap">
            <Slider
              label={`Latency (${(latencyWeight * 100).toFixed(0)}%)`}
              value={Math.round(latencyWeight * 10)}
              min={1} max={9}
              onChange={(v) => setLatencyWeight(v / 10)}
            />
            <p className="weight-rationale">Penalizes prompts that cause higher response times.</p>
          </div>
        </div>
        <div className="weight-total-row">
          <span>Total weights:</span>
          <span className={
            Math.abs(qualityWeight + costWeight + latencyWeight - 1.0) < 0.05
              ? "weight-total weight-total--ok"
              : "weight-total weight-total--warn"
          }>
            {((qualityWeight + costWeight + latencyWeight) * 100).toFixed(0)}%
            {Math.abs(qualityWeight + costWeight + latencyWeight - 1.0) >= 0.05 && " ⚠️ recommended ~100%"}
          </span>
        </div>
      </Card>

      {/* Launch */}
      <Card>
        <Button onClick={handleRun} loading={running} icon={<span className="emoji-icon emoji-rocket">🚀</span>} size="lg">
          Start Optimization
        </Button>
        {error && <div className="alert alert-error mt-3">{error}</div>}
      </Card>

      {/* Results */}
      {result && (
        <>
          <Card accent>
            <SectionHeader title="Optimization Results" />
            <div className="grid-3">
              <Metric label="Baseline Score" value={result.baseline_score.toFixed(3)} />
              <Metric label="Best Score" value={result.best_score.toFixed(3)} />
              <Metric
                label="Improvement"
                value={`${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
                sub={delta > 0 ? "improved" : delta === 0 ? "unchanged" : "degraded"}
                icon={delta > 0 ? <span className="emoji-icon">📈</span> : delta < 0 ? <span className="emoji-icon">📉</span> : <span className="emoji-icon">➖</span>}
              />
            </div>
            <p className="text-muted mt-3">
              Iterations run: <strong>{result.iterations_run}</strong>
            </p>
          </Card>

          {/* History chart */}
          {result.history.length > 0 && (
            <Card>
              <SectionHeader title="Iteration History" />
              <DataTable
                striped
                columns={[
                  { key: "iteration", label: "Iteration", align: "center" },
                  { key: "score", label: "Score", align: "right" },
                  { key: "num_correct", label: "Correct", align: "right" },
                  { key: "baseline", label: "Baseline", align: "center" },
                ]}
                rows={result.history.map((h) => ({
                  iteration: h.iteration,
                  score: h.score.toFixed(3),
                  num_correct: h.num_correct,
                  baseline: h.is_baseline ? "Baseline" : "",
                }))}
              />

              <div className="mt-4">
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart
                    data={result.history.map((h) => ({
                      iter: `Iter ${h.iteration}`,
                      score: h.score,
                    }))}
                  >
                    <defs>
                      <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#39B54A" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#39B54A" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                    <XAxis dataKey="iter" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#39B54A"
                      strokeWidth={2}
                      fill="url(#scoreGrad)"
                      dot={{ r: 5, fill: "#39B54A", strokeWidth: 2, stroke: "#fff" }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </Card>
          )}

          {/* Statistical significance — BN_PROMPT_OPTIMIZATION */}
          {result.statistical_test && (
            <Card accent={result.statistical_test.significant}>
              <SectionHeader title="Significatività Statistica" />
              <div className="stat-test-block">
                <div className="stat-test-row">
                  <span className="stat-test-label">Test</span>
                  <span className="stat-test-val">{result.statistical_test.test_type}</span>
                </div>
                <div className="stat-test-row">
                  <span className="stat-test-label">p-value</span>
                  <span className="stat-test-val">{result.statistical_test.p_value.toFixed(4)}</span>
                </div>
                <div className="stat-test-row">
                  <span className="stat-test-label">Confidence level</span>
                  <span className="stat-test-val">{(result.statistical_test.confidence_level * 100).toFixed(0)}%</span>
                </div>
                <div className="stat-test-row">
                  <span className="stat-test-label">Result</span>
                  <span className={result.statistical_test.significant ? "stat-test-significant" : "stat-test-not-significant"}>
                    {result.statistical_test.significant
                      ? "✅ Statistically significant — the optimal prompt outperforms the baseline"
                      : "⚠️ Not significant — difference is inconclusive"}
                  </span>
                </div>
              </div>
            </Card>
          )}

          {/* Ablation report — BN_PROMPT_OPTIMIZATION */}
          {result.ablation_report && result.ablation_report.length > 0 && (
            <Card>
              <SectionHeader title="Ablation Report" />
              <p className="text-muted mb-3" style={{ fontSize: "0.85rem" }}>
                Contribution of each change to the overall score. Positive values indicate improvement.
              </p>
              <div className="ablation-list">
                {result.ablation_report
                  .slice()
                  .sort((a, b) => b.delta_score - a.delta_score)
                  .map((entry, i) => (
                    <div key={i} className="ablation-row">
                      <span className="ablation-iter">Iter {entry.iteration}</span>
                      <span className="ablation-change">{entry.change}</span>
                      <span className={entry.delta_score >= 0 ? "ablation-delta ablation-delta--pos" : "ablation-delta ablation-delta--neg"}>
                        {entry.delta_score >= 0 ? "+" : ""}{entry.delta_score.toFixed(3)}
                      </span>
                    </div>
                  ))}
              </div>
            </Card>
          )}

          {/* Best prompt */}
          {result.best_prompt && (
            <Card>
              <SectionHeader title="Best Prompt Found" />
              <pre className="code-block">{result.best_prompt.slice(0, 3000)}</pre>
            </Card>
          )}

          {/* All variants */}
          {result.all_variants.length > 0 && (
            <Card>
              <button
                className="collapse-trigger full-width"
                onClick={() => setExpandVariants(!expandVariants)}
                type="button"
              >
                {expandVariants ? "▼" : "▶"} All variants ({result.all_variants.length})
              </button>
              {expandVariants && (
                <div className="mt-3 fade-in">
                  {result.all_variants.map((v, i) => (
                    <div key={i} className="variant-card mb-3">
                      <div className="flex-between">
                        <strong>Iter {v.iteration}, Var {v.variant_index}</strong>
                        <span className="badge badge-info">Score: {v.score.toFixed(3)}</span>
                      </div>
                      <pre className="code-block mt-2">{v.prompt.slice(0, 500)}</pre>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
