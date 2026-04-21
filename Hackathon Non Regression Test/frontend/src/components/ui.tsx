import { clsx } from "clsx";
import { Loader2, Check, ChevronDown, Inbox } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import type { ReactNode, ButtonHTMLAttributes } from "react";

// --- Card ---

export function Card({
  children,
  className,
  accent,
}: {
  children: ReactNode;
  className?: string;
  accent?: boolean;
}) {
  return (
    <div className={clsx("card", accent && "card-accent", className)}>
      {children}
    </div>
  );
}

// --- Badge ---

export function Badge({
  children,
  variant = "default",
}: {
  children: ReactNode;
  variant?: "default" | "success" | "warning" | "error" | "info";
}) {
  return <span className={clsx("badge", `badge-${variant}`)}>{children}</span>;
}

// --- Metric card ---

export function Metric({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">{label}</span>
        {icon && <span className="metric-icon">{icon}</span>}
      </div>
      <span className="metric-value">{value}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}

// --- Button ---

interface BtnProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  loading,
  icon,
  children,
  className,
  disabled,
  ...rest
}: BtnProps) {
  return (
    <button
      className={clsx("btn", `btn-${variant}`, `btn-${size}`, className)}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 size={16} className="spin" /> : icon}
      {children}
    </button>
  );
}

// --- Toggle Card (replaces MultiCheck for model/task selection) ---

export function ToggleCard({
  label,
  description,
  selected,
  onClick,
  icon,
}: {
  label: string;
  description?: string;
  selected: boolean;
  onClick: () => void;
  icon?: ReactNode;
}) {
  return (
    <button
      type="button"
      className={clsx("toggle-card", selected && "toggle-card-active")}
      onClick={onClick}
    >
      <div className="toggle-card-content">
        {icon && <span className="toggle-card-icon">{icon}</span>}
        <div className="toggle-card-text">
          <span className="toggle-card-label">{label}</span>
          {description && (
            <span className="toggle-card-desc">{description}</span>
          )}
        </div>
      </div>
      <div className={clsx("toggle-indicator", selected && "toggle-indicator-active")}>
        {selected && <Check size={12} />}
      </div>
    </button>
  );
}

// --- Multi-select toggle group ---

export function MultiToggle({
  label,
  options,
  selected,
  onChange,
}: {
  label?: string;
  options: { value: string; label: string; description?: string }[];
  selected: string[];
  onChange: (selected: string[]) => void;
}) {
  const toggle = (val: string) => {
    onChange(
      selected.includes(val)
        ? selected.filter((x) => x !== val)
        : [...selected, val]
    );
  };

  const allSelected = options.length > 0 && options.every((o) => selected.includes(o.value));

  return (
    <div className="field">
      {label && (
        <div className="field-header">
          <span className="field-label">{label}</span>
          <button
            type="button"
            className="select-all-btn"
            onClick={() =>
              onChange(allSelected ? [] : options.map((o) => o.value))
            }
          >
            {allSelected ? "Deselect all" : "Select all"}
          </button>
        </div>
      )}
      <div className="toggle-grid">
        {options.map((o) => (
          <ToggleCard
            key={o.value}
            label={o.label}
            description={o.description}
            selected={selected.includes(o.value)}
            onClick={() => toggle(o.value)}
          />
        ))}
      </div>
    </div>
  );
}

// --- Legacy MultiCheck (kept for chip-style selections) ---
export function MultiCheck({
  label,
  options,
  selected,
  onChange,
}: {
  label?: string;
  options: { value: string; label: string }[];
  selected: string[];
  onChange: (selected: string[]) => void;
}) {
  const toggle = (val: string) => {
    onChange(
      selected.includes(val)
        ? selected.filter((x) => x !== val)
        : [...selected, val]
    );
  };

  return (
    <div className="field">
      {label && <span className="field-label">{label}</span>}
      <div className="chip-group">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className={clsx("chip", selected.includes(o.value) && "chip-active")}
            onClick={() => toggle(o.value)}
          >
            {selected.includes(o.value) && <Check size={12} />}
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// --- Custom Select ---

export function SelectField({
  label,
  value,
  options,
  onChange,
  className,
}: {
  label?: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // Elevate parent .card z-index when dropdown is open
  useEffect(() => {
    const card = ref.current?.closest(".card") as HTMLElement | null;
    if (card) {
      card.style.zIndex = open ? "100" : "";
    }
  }, [open]);

  const selected = options.find((o) => o.value === value);

  return (
    <div className={clsx("field", className)} ref={ref} style={{ position: "relative", zIndex: open ? 999 : undefined }}>
      {label && <span className="field-label">{label}</span>}
      <div className="custom-select-wrapper">
        <button
          type="button"
          className={clsx("custom-select-trigger", open && "custom-select-trigger--open")}
          onClick={() => setOpen((v) => !v)}
        >
          <span className="custom-select-value">{selected?.label ?? "—"}</span>
          <ChevronDown size={16} className={clsx("custom-select-chevron", open && "custom-select-chevron--open")} />
        </button>

        <div className={clsx("custom-select-dropdown", open && "custom-select-dropdown--open")}>
          {options.map((o) => (
            <button
              type="button"
              key={o.value}
              className={clsx("custom-select-option", o.value === value && "custom-select-option--active")}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              <span>{o.label}</span>
              {o.value === value && <Check size={14} className="custom-select-check" />}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// --- Slider with track fill ---

export function Slider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="field">
      <div className="slider-header">
        <span className="field-label">{label}</span>
        <span className="slider-value">{value}</span>
      </div>
      <div className="slider-track-wrap">
        <div className="slider-track">
          <div className="slider-track-fill" style={{ width: `${pct}%` }} />
        </div>
        <input
          type="range"
          className="slider"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <div className="slider-thumb-glow" style={{ left: `${pct}%` }} />
      </div>
    </label>
  );
}

// --- Status badge ---

const STATUS_MAP: Record<string, { label: string; variant: string }> = {
  pending: { label: "Pending", variant: "warning" },
  running: { label: "Running", variant: "info" },
  completed: { label: "Completed", variant: "success" },
  failed: { label: "Failed", variant: "error" },
};

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_MAP[status] || { label: status, variant: "default" };
  return (
    <span className={clsx("status-pill", `status-${s.variant}`)}>
      <span className="status-dot" />
      {s.label}
    </span>
  );
}

// --- Empty state ---

export function Empty({ message }: { message: string }) {
  return (
    <div className="empty-state">
      <Inbox size={40} strokeWidth={1.2} />
      <p>{message}</p>
    </div>
  );
}

// --- Spinner ---

export function Spinner({ text }: { text?: string }) {
  return (
    <div className="spinner-overlay">
      <div className="spinner-ring" />
      {text && <p>{text}</p>}
    </div>
  );
}

// --- Section header ---

export function SectionHeader({
  step,
  title,
  right,
}: {
  step?: number | string;
  title: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="section-header">
      <div className="section-header-left">
        {step !== undefined && <span className="step-badge">{step}</span>}
        <h3>{title}</h3>
      </div>
      {right && <div className="section-header-right">{right}</div>}
    </div>
  );
}

// --- Data Table ---

export function DataTable({
  columns,
  rows,
  maxHeight,
  striped,
}: {
  columns: {
    key: string;
    label: string;
    align?: "left" | "center" | "right";
    render?: (value: unknown, row: Record<string, unknown>) => ReactNode;
  }[];
  rows: Record<string, unknown>[];
  maxHeight?: number;
  striped?: boolean;
}) {
  if (rows.length === 0) {
    return (
      <div className="table-empty">
        <Inbox size={24} strokeWidth={1.5} />
        <span>No data</span>
      </div>
    );
  }

  const AUTO_SCROLL_THRESHOLD = 6;
  const effectiveMaxHeight =
    maxHeight ?? (rows.length > AUTO_SCROLL_THRESHOLD ? 420 : undefined);

  return (
    <div className="table-wrap" style={effectiveMaxHeight ? { maxHeight: effectiveMaxHeight } : undefined}>
      <table className={clsx("data-table", striped && "data-table-striped")}>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} style={{ textAlign: c.align || "left" }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key} style={{ textAlign: c.align || "left" }}>
                  {c.render
                    ? c.render(row[c.key], row)
                    : String(row[c.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- Progress bar ---

export function ProgressBar({
  value,
  max = 1,
  label,
}: {
  value: number;
  max?: number;
  label?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="progress-bar-wrap">
      {label && <span className="progress-label">{label}</span>}
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
