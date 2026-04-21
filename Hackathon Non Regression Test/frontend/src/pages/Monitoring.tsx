import { api } from "../api";
import type { JobSummary } from "../api";
import { usePolling } from "../hooks";
import { Card, Empty, Metric, StatusBadge } from "../components/ui";
import { useState } from "react";
import { ChevronDown } from "lucide-react";

export default function MonitoringPage() {
  /**
   * BN_ADVANCED_FRONTEND — Auto-polling every 3s (≤ 5s).
   * Job list refreshes automatically so running/completed/failed status
   * updates in real-time without manual page reload.
   */
  const jobs = usePolling<{ jobs: JobSummary[] }>(api.listJobs, 3000, true);
  const [expanded, setExpanded] = useState<string | null>(null);

  const list = jobs?.jobs || [];
  const running = list.filter((j) => j.status === "running").length;
  const completed = list.filter((j) => j.status === "completed").length;
  const failed = list.filter((j) => j.status === "failed").length;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Job Monitoring</h2>
        <p className="page-subtitle">Real-time status of all pipeline jobs</p>
      </div>

      {list.length === 0 ? (
        <Empty message="No jobs started. Go to Config Job to begin." />
      ) : (
        <>
          {/* Overview stats */}
          <div className="grid-4">
            <Metric label="Total" value={list.length} icon={<span className="icon-anim icon-pulse"><span className="emoji-icon">📦</span></span>} />
            <Metric label="Running" value={running} icon={<span className="icon-anim icon-spin-slow"><span className="emoji-icon">⏳</span></span>} />
            <Metric label="Completed" value={completed} icon={<span className="icon-anim icon-bounce"><span className="emoji-icon">✅</span></span>} />
            <Metric label="Failed" value={failed} icon={<span className="icon-anim"><span className="emoji-icon">❌</span></span>} />
          </div>

          {/* Job cards */}
          <div className="job-list">
            {list.map((j) => {
              const isOpen = expanded === j.job_id;
              return (
                <Card key={j.job_id} className={`job-card ${isOpen ? "job-card-open" : ""}`}>
                  <button
                    type="button"
                    className="job-card-header"
                    onClick={() => setExpanded(isOpen ? null : j.job_id)}
                  >
                    <div className="job-card-left">
                      <span className="job-card-id">{j.job_id}</span>
                      <span className="job-card-dataset">{j.dataset_key}</span>
                    </div>
                    <div className="job-card-right">
                      <StatusBadge status={j.status} />
                      <span className="job-card-time">{j.created_at?.slice(0, 16).replace("T", " ")}</span>
                      <ChevronDown size={16} className={`chevron ${isOpen ? "chevron-open" : ""}`} />
                    </div>
                  </button>

                  {isOpen && (
                    <div className="job-card-body fade-in">
                      <div className="grid-3">
                        <Metric label="Tasks" value={j.selected_tasks.length} />
                        <Metric label="Models" value={j.selected_models.length} />
                        <Metric label="Records" value={j.num_records ?? "—"} />
                      </div>

                      {j.selected_tasks.length > 0 && (
                        <div className="job-tags mt-3">
                          {j.selected_tasks.map((t) => (
                            <span key={t} className="job-tag">{t}</span>
                          ))}
                        </div>
                      )}

                      {j.error && (
                        <div className="alert alert-error mt-3">{j.error}</div>
                      )}

                      {j.synthetic_count != null && (
                        <p className="text-muted mt-2">Synthetic records: <strong>{j.synthetic_count}</strong></p>
                      )}

                      {j.completed_at && (
                        <p className="text-muted mt-2">Completed: {j.completed_at.slice(0, 19).replace("T", " ")}</p>
                      )}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
