import { useState, useEffect } from "react";
import { api } from "../api";
import type { AgentInfo, ModelConfig } from "../api";
import {
  Badge,
  Button,
  Card,
  DataTable,
  Empty,
  Metric,
  MultiToggle,
  SectionHeader,
  Slider,
  Spinner,
} from "../components/ui";

export default function ConfigJobPage() {
  const [bucket, setBucket] = useState("");
  const [datasets, setDatasets] = useState<string[]>([]);
  const [datasetKey, setDatasetKey] = useState("dataset/dataset.jsonl");

  const [detecting, setDetecting] = useState(false);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [availableTasks, setAvailableTasks] = useState<string[]>([]);
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [totalRecords, setTotalRecords] = useState(0);

  const [models, setModels] = useState<ModelConfig[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  const [enrichSynthetic, setEnrichSynthetic] = useState(false);
  const [synthPerTask, setSynthPerTask] = useState(5);

  const [launching, setLaunching] = useState(false);
  const [launchResult, setLaunchResult] = useState<{
    ok: boolean;
    msg: string;
  } | null>(null);

  useEffect(() => {
    api.getDatasets().then((d) => {
      setDatasets(d.datasets);
      setBucket(d.bucket);
      if (d.datasets.length > 0) setDatasetKey(d.datasets[0]);
    }).catch(() => {});

    api.getModels().then((d) => {
      setModels(d.models);
      setSelectedModels(d.models.map((m) => m.model_id));
    }).catch(() => {});

    api.getDetected().then((d) => {
      if (d.detected) {
        setAgents(d.agents);
        setAvailableTasks(d.available_tasks);
        setSelectedTasks(d.available_tasks);
        setDatasetKey(d.dataset_key);
        setTotalRecords(d.total_records);
      }
    }).catch(() => {});
  }, []);

  const handleDetect = async () => {
    setDetecting(true);
    setLaunchResult(null);
    try {
      const result = await api.detectTasks(bucket, datasetKey);
      setAgents(result.agents);
      setAvailableTasks(result.available_tasks);
      setSelectedTasks(result.available_tasks);
      setTotalRecords(result.total_records);
    } catch (e) {
      setLaunchResult({
        ok: false,
        msg: `Detection error: ${e instanceof Error ? e.message : e}`,
      });
    } finally {
      setDetecting(false);
    }
  };

  const handleLaunch = async () => {
    setLaunching(true);
    setLaunchResult(null);
    try {
      const res = await api.createJob({
        dataset_key: datasetKey,
        selected_tasks: selectedTasks,
        selected_models: selectedModels,
        enrich_synthetic: enrichSynthetic,
        synth_records_per_task: synthPerTask,
      });
      setLaunchResult({
        ok: true,
        msg: `Job ${res.job_id} started! Go to Monitoring to track its status.`,
      });
    } catch (e) {
      setLaunchResult({
        ok: false,
        msg: `Error: ${e instanceof Error ? e.message : e}`,
      });
    } finally {
      setLaunching(false);
    }
  };

  const canLaunch =
    agents.length > 0 && selectedTasks.length > 0 && selectedModels.length > 0;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Job Configuration</h2>
        <p className="page-subtitle">Configure and launch the NRT evaluation pipeline</p>
      </div>

      {/* Step 1 */}
      <Card>
        <SectionHeader step={1} title="Detect Tasks" />
        <Button onClick={handleDetect} loading={detecting} icon={<span className="emoji-icon emoji-search">🔎</span>}>
          Detect Tasks
        </Button>

        {detecting && <Spinner text="Loading dataset and detecting tasks..." />}

        {agents.length > 0 && !detecting && (
          <div className="mt-4 fade-in">
            <div className="grid-3 mb-4">
              <Metric label="Agents" value={agents.length} icon={<span className="icon-anim icon-bounce"><span className="emoji-icon">🤖</span></span>} />
              <Metric label="Distinct Tasks" value={availableTasks.length} icon={<span className="icon-anim icon-pulse"><span className="emoji-icon">📋</span></span>} />
              <Metric label="Total Records" value={totalRecords.toLocaleString()} icon={<span className="icon-anim icon-wiggle"><span className="emoji-icon">💾</span></span>} />
            </div>

            <div className="agent-table-wrap">
              <table className="agent-table">
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>Agent</th>
                    <th style={{ textAlign: "left" }}>Task</th>
                    <th style={{ textAlign: "right" }}>Records</th>
                  </tr>
                </thead>
                <tbody>
                  {agents.map((a, i) => {
                    const maxRec = Math.max(...agents.map((x) => x.num_records));
                    const pct = maxRec > 0 ? (a.num_records / maxRec) * 100 : 0;
                    return (
                      <tr key={a.agent_id} style={{ animationDelay: `${i * 40}ms` }}>
                        <td>
                          <div className="agent-cell">
                            <span className="agent-avatar">
                              <span>🤖</span>
                            </span>
                            <span className="agent-name">{a.agent_id}</span>
                          </div>
                        </td>
                        <td><Badge variant="info">{a.task}</Badge></td>
                        <td style={{ textAlign: "right" }}>
                          <div className="records-cell">
                            <div className="records-bar-bg">
                              <div className="records-bar-fill" style={{ width: `${pct}%` }} />
                            </div>
                            <span className="records-num">{a.num_records.toLocaleString()}</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>

      {/* Step 2 */}
      {agents.length > 0 && (
        <Card className="fade-in">
          <SectionHeader step={2} title="Select Tasks & Models" />

          <MultiToggle
            label="Tasks to evaluate"
            options={availableTasks.map((t) => ({
              value: t,
              label: t.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
              description: `${agents.filter((a) => a.task === t).reduce((s, a) => s + a.num_records, 0)} records`,
            }))}
            selected={selectedTasks}
            onChange={setSelectedTasks}
          />

          <div className="mt-4">
            <MultiToggle
              label="Candidate Models"
              options={models.map((m) => ({
                value: m.model_id,
                label: m.display_name || m.model_id,
                description: m.provider || "",
              }))}
              selected={selectedModels}
              onChange={setSelectedModels}
            />
          </div>
        </Card>
      )}

      {/* Steps 4 & 5 side by side */}
      {agents.length > 0 && (
        <div className="grid-2 grid-2-stretch">
          <Card className="fade-in card-compact">
            <SectionHeader step={3} title="Synthetic Dataset" />

            <div className="synth-toggle-row">
              <div className="synth-toggle-label">
                <span className="emoji-icon">🧬</span>
                <span>Synthetic generation</span>
              </div>
              <button
                type="button"
                className={`mini-switch ${enrichSynthetic ? "mini-switch-on" : ""}`}
                onClick={() => setEnrichSynthetic(!enrichSynthetic)}
              >
                <span className="mini-switch-thumb" />
              </button>
            </div>

            {enrichSynthetic && (
              <div className="synth-config fade-in">
                <Slider
                  label="Records per task"
                  value={synthPerTask}
                  min={1}
                  max={30}
                  onChange={setSynthPerTask}
                />
                <div className="synth-summary synth-summary-sm">
                  <div className="synth-summary-item">
                    <span className="synth-summary-num">{synthPerTask}</span>
                    <span className="synth-summary-label">per task</span>
                  </div>
                  <span className="synth-summary-x">&times;</span>
                  <div className="synth-summary-item">
                    <span className="synth-summary-num">{selectedTasks.length}</span>
                    <span className="synth-summary-label">tasks</span>
                  </div>
                  <span className="synth-summary-eq">=</span>
                  <div className="synth-summary-item synth-summary-total">
                    <span className="synth-summary-num">{synthPerTask * selectedTasks.length}</span>
                    <span className="synth-summary-label">total</span>
                  </div>
                </div>
              </div>
            )}
          </Card>

          <Card className="card-compact">
            <SectionHeader step={4} title="Launch Pipeline" />

            {canLaunch && (
              <div className="launch-summary mb-3">
                <span>{selectedTasks.length} tasks</span>
                <span className="launch-dot" />
                <span>{selectedModels.length} models</span>
                <span className="launch-dot" />
                <span>{enrichSynthetic ? "Synth ON" : "Synth OFF"}</span>
              </div>
            )}

            <Button
              onClick={handleLaunch}
              loading={launching}
              disabled={!canLaunch}
              icon={<span className="emoji-icon emoji-rocket">🚀</span>}
            >
              Launch Pipeline
            </Button>

            {launchResult && (
              <div className={`alert mt-3 ${launchResult.ok ? "alert-success" : "alert-error"}`}>
                {launchResult.msg}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
