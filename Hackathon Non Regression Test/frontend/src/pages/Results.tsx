import { useState, useEffect, useRef } from "react";
import { api } from "../api";
import type { JobSummary, JobDetail } from "../api";
import {
  Card,
  DataTable,
  Empty,
  Metric,
  ProgressBar,
  SelectField,
  SectionHeader,
  StatusBadge,
} from "../components/ui";

/* ─── AI typewriter hook ─── */
function useTypewriter(text: string, speed = 18) {
  const [display, setDisplay] = useState("");
  const [done, setDone] = useState(false);
  const idx = useRef(0);

  useEffect(() => {
    idx.current = 0;
    setDisplay("");
    setDone(false);
    if (!text) return;
    const iv = setInterval(() => {
      idx.current++;
      setDisplay(text.slice(0, idx.current));
      if (idx.current >= text.length) {
        clearInterval(iv);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(iv);
  }, [text, speed]);

  return { display, done };
}

/* ─── Intersection Observer hook ─── */
function useInView(opts?: IntersectionObserverInit) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVisible(true); obs.disconnect(); }
    }, { threshold: 0.3, ...opts });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return { ref, visible };
}

/* ─── Insight block with typewriter ─── */
function InsightBlock({ summary, recommendation }: { summary?: string; recommendation?: string }) {
  const { ref, visible } = useInView();
  const summaryTw = useTypewriter(visible ? (summary || "") : "", 14);
  const recoTw = useTypewriter(visible ? (recommendation || "") : "", 14);
  const allDone = summaryTw.done && (!recommendation || recoTw.done);

  return (
    <div ref={ref} className="insight-ai-block">
      <div className="insight-ai-header">
        <span className="emoji-icon emoji-robot" role="img" aria-label="robot">🤖</span>
        <span className="insight-ai-label">AI Insight</span>
        {visible && !allDone && <span className="insight-ai-status">analyzing…</span>}
        {allDone && <span className="insight-ai-status insight-ai-done">analysis complete ✓</span>}
      </div>

      {visible && (
        <div className="insight-ai-body">
          {summary && (
            <div className="insight-ai-bubble insight-ai-summary">
              <span className="insight-ai-bubble-label"><span className="emoji-icon emoji-sparkle">✨</span> Summary</span>
              <p>{summaryTw.display}<span className={`typing-cursor ${summaryTw.done ? "hidden" : ""}`}>▌</span></p>
            </div>
          )}
          {recommendation && (
            <div className="insight-ai-bubble insight-ai-reco">
              <span className="insight-ai-bubble-label"><span className="emoji-icon emoji-rocket">🚀</span> Recommendation</span>
              <p>{recoTw.display}<span className={`typing-cursor ${recoTw.done ? "hidden" : ""}`}>▌</span></p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── Task Detail Tabs ─── */
function TaskDetailTabs({ tasks, perTaskModelDict }: { tasks: string[]; perTaskModelDict: Record<string, any> }) {
  const [active, setActive] = useState(tasks[0] || "");

  const rows = Object.values(perTaskModelDict)
    .filter((v: any) => v.task === active)
    .map((v: any) => ({
      model: v.model_id,
      score: v.avg_score,
      scoreDisplay: v.avg_score?.toFixed(3),
      accuracy: v.accuracy,
      accuracyDisplay: (v.accuracy * 100).toFixed(1) + "%",
      records: v.num_records,
    }))
    .sort((a, b) => b.score - a.score);

  return (
    <Card>
      <SectionHeader title={<><span className="emoji-icon emoji-search" role="img" aria-label="detail">🔎</span> Task Details</>} />

      <div className="task-tabs">
        {tasks.map((t) => (
          <button
            key={t}
            className={`task-tab ${t === active ? "active" : ""}`}
            onClick={() => setActive(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="task-detail-grid" key={active}>
        {rows.map((r, i) => (
          <div key={r.model} className="task-detail-card" style={{ animationDelay: `${i * 80}ms` }}>
            <div className="task-detail-body">
              <span className="task-detail-model">{r.model}</span>
              <div className="task-detail-bar">
                <ProgressBar value={r.score} />
              </div>
              <div className="task-detail-nums">
                <span className="task-detail-score">{r.scoreDisplay}</span>
                <span className="task-detail-acc">{r.accuracyDisplay} acc</span>
                <span className="task-detail-rec">{r.records} rec</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function ResultsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [detail, setDetail] = useState<JobDetail | null>(null);

  /**
   * BN_ADVANCED_FRONTEND — Auto-polling every 4s (≤ 5s).
   * Refreshes the job list so newly completed jobs appear automatically.
   * Also re-fetches the selected job detail if it's still running.
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
        if (completed.length > 0 && !selectedId) {
          setSelectedId(completed[0].job_id);
        }
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
    // Poll detail while job is still running
    const id = setInterval(fetchDetail, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [selectedId]);

  if (!jobs.length) {
    return (
      <div className="page">
        <div className="page-header">
          <h2 className="page-title">Results</h2>
        </div>
        <Empty message="No completed jobs. Launch a pipeline from Config Job." />
      </div>
    );
  }

  const agg = detail?.aggregated;

  const perModelRows = agg?.per_model
    ? Object.entries(agg.per_model)
        .map(([mid, info]) => ({
          model: mid,
          score: info.overall_score,
          scoreDisplay: info.overall_score?.toFixed(3),
          accuracy: info.accuracy,
          accuracyDisplay: (info.accuracy * 100).toFixed(1) + "%",
          records: info.num_records,
          correct: info.num_correct,
        }))
        .sort((a, b) => b.score - a.score)
    : [];

  const bestOverall = agg?.best_models?.overall;
  const bestPerTask = agg?.best_models?.per_task;

  const perTaskModelDict = agg?.per_task_model || {};
  const tasks = [
    ...new Set(Object.values(perTaskModelDict).map((v) => v.task)),
  ].sort();

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Results</h2>
        <p className="page-subtitle">Performance analysis of evaluated models</p>
      </div>

      {/* Job selector */}
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

      {!agg && detail && (
        <Card>
          <Empty message="Aggregated data not available for this job." />
        </Card>
      )}

      {/* Best model highlight */}
      {bestOverall?.model_id && (
        <div className="best-model-hero">
          <div className="best-model-hero-left">
            <span className="emoji-icon emoji-trophy" role="img" aria-label="trophy">🏆</span>
            <div>
              <h3 className="best-model-title">Best Overall Model</h3>
              <p className="best-model-name">{bestOverall.model_id}</p>
            </div>
          </div>
          <div className="best-model-hero-score">
            <span className="best-model-score-num">{bestOverall.overall_score?.toFixed(3)}</span>
            <span className="best-model-score-label">score</span>
          </div>
        </div>
      )}

      {/* Model metrics */}
      {perModelRows.length > 0 && (
        <Card>
          <SectionHeader
            title="Model Metrics"
            right={<span className="text-muted">{perModelRows.length} models</span>}
          />

          <div className="model-ranking">
            {perModelRows.map((r, i) => (
              <div key={r.model} className="model-rank-row">
                <div className="model-rank-position">
                  {i === 0 ? <span className="emoji-icon emoji-medal" role="img" aria-label="gold">🥇</span>
                   : i === 1 ? <span className="emoji-icon emoji-medal" role="img" aria-label="silver">🥈</span>
                   : i === 2 ? <span className="emoji-icon emoji-medal" role="img" aria-label="bronze">🥉</span>
                   : <span className="rank-num">#{i + 1}</span>}
                </div>
                <div className="model-rank-info">
                  <span className="model-rank-name">{r.model}</span>
                  <div className="model-rank-stats">
                    <span>{r.records} records</span>
                    <span>{r.correct} correct</span>
                  </div>
                </div>
                <div className="model-rank-metrics">
                  <div className="model-rank-score">
                    <span className="score-num">{r.scoreDisplay}</span>
                    <ProgressBar value={r.score} />
                  </div>
                  <span className="model-rank-accuracy">{r.accuracyDisplay}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Best per task */}
      {bestPerTask && Object.keys(bestPerTask).length > 0 && (
        <Card>
          <SectionHeader title={<><span className="emoji-icon emoji-star" role="img" aria-label="star">⭐</span> Best Model per Task</>} />
          <div className="model-ranking">
            {Object.entries(bestPerTask)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([task, info], i) => (
                <div key={task} className="model-rank-row">
                  <div className="model-rank-position">
                    <span className="task-rank-badge">{i + 1}</span>
                  </div>
                  <div className="model-rank-info">
                    <span className="model-rank-name">{task}</span>
                    <div className="model-rank-stats">
                      <span>🏅 {info.model_id}</span>
                    </div>
                  </div>
                  <div className="model-rank-metrics">
                    <div className="model-rank-score">
                      <span className="score-num">{info.avg_score?.toFixed(3)}</span>
                      <ProgressBar value={info.avg_score || 0} />
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </Card>
      )}

      {/* Per task detail — tabbed */}
      {tasks.length > 0 && (
        <TaskDetailTabs tasks={tasks} perTaskModelDict={perTaskModelDict} />
      )}

      {/* Insights — AI typing */}
      {detail?.insights?.insights && (
        <Card>
          <InsightBlock
            summary={detail.insights.insights.summary}
            recommendation={detail.insights.insights.recommendation}
          />
        </Card>
      )}

      {/* Download */}
      {detail && (detail.has_report || detail.has_json) && (
        <Card>
          <SectionHeader title={<><span className="emoji-icon emoji-download" role="img" aria-label="download">📥</span> Download Report</>} />
          <div className="download-group">
            {detail.has_report && (
              <a href={api.getReportUrl(selectedId, "xlsx")} className="download-card" download>
                <span className="emoji-icon emoji-download">📄</span>
                <div>
                  <span className="download-title">Excel Report</span>
                  <span className="download-desc">.xlsx with all details</span>
                </div>
              </a>
            )}
            {detail.has_json && (
              <a href={api.getReportUrl(selectedId, "json")} className="download-card" download>
                <span className="emoji-icon emoji-download">📂</span>
                <div>
                  <span className="download-title">JSON Report</span>
                  <span className="download-desc">Structured data</span>
                </div>
              </a>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
