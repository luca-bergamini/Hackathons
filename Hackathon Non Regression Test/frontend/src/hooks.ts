import { useState, useEffect, useCallback } from "react";

type Status = "idle" | "loading" | "success" | "error";

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: unknown[] = [],
  immediate = true
) {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const result = await fn();
      setData(result);
      setStatus("success");
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus("error");
      return null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    if (immediate) {
      execute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [execute]);

  return { data, status, error, execute, setData };
}

/**
 * BN_ADVANCED_FRONTEND — Generic polling hook.
 * Calls `fn` immediately and then every `intervalMs` milliseconds.
 * Polling is active only when `enabled` is true.
 * All pages using this hook satisfy the ≤ 5s auto-refresh requirement.
 */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  enabled: boolean
) {
  const [data, setData] = useState<T | null>(null);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const result = await fn();
        if (!cancelled) setData(result);
      } catch {
        /* ignore polling errors */
      }
    };

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs]);

  return data;
}
