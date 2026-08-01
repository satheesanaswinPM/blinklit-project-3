import { useEffect, useMemo, useState } from "react";

const API = import.meta.env.VITE_API_BASE || "/api";

const INSIGHT_TABS = [
  { id: "", label: "All themes" },
  { id: "habit_drivers", label: "Habit drivers" },
  { id: "barrier_taxonomy", label: "Barriers" },
  { id: "discovery_path_map", label: "Discovery paths" },
  { id: "info_needs", label: "Info needs" },
];

async function fetchJson(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function sentimentSummary(mix) {
  const entries = Object.entries(mix || {});
  if (!entries.length) return "—";
  return entries.map(([k, v]) => `${k} ${v}`).join(" · ");
}

export default function App() {
  const [stats, setStats] = useState(null);
  const [themes, setThemes] = useState([]);
  const [overview, setOverview] = useState(null);
  const [insightType, setInsightType] = useState("");
  const [barrier, setBarrier] = useState("");
  const [source, setSource] = useState("");
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [s, o] = await Promise.all([
          fetchJson("/stats"),
          fetchJson("/insights/overview"),
        ]);
        if (!cancelled) {
          setStats(s);
          setOverview(o);
        }
      } catch (e) {
        if (!cancelled) setError(String(e.message || e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams();
        if (insightType) params.set("insight_type", insightType);
        if (barrier) params.set("barrier", barrier);
        if (source) params.set("source", source);
        if (q.trim()) params.set("q", q.trim());
        const data = await fetchJson(`/themes?${params.toString()}`);
        if (!cancelled) {
          setThemes(data.themes || []);
          setSelected((prev) => {
            if (prev && data.themes?.some((t) => t.id === prev.id)) {
              return data.themes.find((t) => t.id === prev.id);
            }
            return data.themes?.[0] || null;
          });
          setError("");
        }
      } catch (e) {
        if (!cancelled) setError(String(e.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [insightType, barrier, source, q]);

  const barrierOptions = useMemo(() => {
    const keys = Object.keys(stats?.themes_by_barrier || {});
    return keys.sort();
  }, [stats]);

  const sourceOptions = useMemo(() => {
    return Object.keys(stats?.by_source || {}).sort();
  }, [stats]);

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="eyebrow">Phase 1 · Discovery Insight Engine</p>
          <h1>Theme Explorer</h1>
          <p className="sub">
            Multi-channel feedback themes with grounded evidence quotes — habit,
            barriers, discovery paths, and info needs.
          </p>
        </div>
        <div className="meta">
          <div className="stat">
            <span className="stat-label">Docs</span>
            <span className="stat-value">{stats?.n_docs ?? "—"}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Themes</span>
            <span className="stat-value">{stats?.n_themes ?? "—"}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Run</span>
            <span className="stat-value mono">
              {stats?.latest_run?.id?.slice(0, 12) ?? "—"}
            </span>
          </div>
        </div>
      </header>

      {error && (
        <div className="banner">
          Could not reach API ({error}). Start uvicorn on port 8000, then refresh.
        </div>
      )}

      <section className="overview">
        <h2>Research questions at a glance</h2>
        <div className="overview-grid">
          {[
            ["Habit drivers", overview?.habit_drivers],
            ["Barriers", overview?.barriers],
            ["Discovery paths", overview?.discovery_paths],
            ["Info needs", overview?.info_needs],
          ].map(([label, items]) => (
            <div key={label} className="overview-card">
              <h3>{label}</h3>
              <ul>
                {(items || []).slice(0, 3).map((t) => (
                  <li key={t.id}>
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => {
                        setInsightType("");
                        setSelected(null);
                        setQ(t.title.split(":")[0]);
                      }}
                    >
                      {t.title}
                    </button>
                    <span className="vol">n={t.volume}</span>
                  </li>
                ))}
                {!items?.length && <li className="muted">No themes yet — run the pipeline.</li>}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <div className="toolbar">
        <div className="tabs">
          {INSIGHT_TABS.map((tab) => (
            <button
              key={tab.id || "all"}
              type="button"
              className={insightType === tab.id ? "tab active" : "tab"}
              onClick={() => setInsightType(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="filters">
          <select value={barrier} onChange={(e) => setBarrier(e.target.value)}>
            <option value="">All barriers</option>
            {barrierOptions.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {sourceOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <input
            type="search"
            placeholder="Search themes or quotes"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>

      <div className="workspace">
        <aside className="list">
          {loading && <p className="muted">Loading themes…</p>}
          {!loading && themes.length === 0 && (
            <p className="muted">No themes match these filters.</p>
          )}
          {themes.map((t) => (
            <button
              key={t.id}
              type="button"
              className={selected?.id === t.id ? "theme-row active" : "theme-row"}
              onClick={() => setSelected(t)}
            >
              <span className="theme-title">{t.title}</span>
              <span className="theme-meta">
                vol {t.volume} · {t.barriers?.[0] || "—"}
              </span>
            </button>
          ))}
        </aside>

        <main className="detail">
          {!selected && <p className="muted">Select a theme to inspect evidence.</p>}
          {selected && (
            <>
              <h2>{selected.title}</h2>
              <p className="detail-sub">
                Volume {selected.volume} · Sentiment {sentimentSummary(selected.sentiment_mix)} ·
                Coherence {Number(selected.coherence || 0).toFixed(2)}
              </p>
              <div className="chips">
                {(selected.insight_types || []).map((x) => (
                  <span key={x} className="chip">
                    {x}
                  </span>
                ))}
                {(selected.barriers || []).map((x) => (
                  <span key={x} className="chip chip-alt">
                    {x}
                  </span>
                ))}
                {(selected.categories || []).map((x) => (
                  <span key={x} className="chip chip-cat">
                    {x}
                  </span>
                ))}
              </div>

              {(selected.discovery_paths?.length > 0 || selected.info_needs?.length > 0) && (
                <div className="extra">
                  {selected.discovery_paths?.length > 0 && (
                    <p>
                      <strong>Discovery paths:</strong> {selected.discovery_paths.join(", ")}
                    </p>
                  )}
                  {selected.info_needs?.length > 0 && (
                    <p>
                      <strong>Info needs:</strong> {selected.info_needs.join(", ")}
                    </p>
                  )}
                </div>
              )}

              <h3>Evidence quotes</h3>
              <p className="muted small">
                Every quote is a grounded span from a source document (no ungrounded claims).
              </p>
              <ul className="quotes">
                {(selected.evidence || []).map((e) => (
                  <li key={`${e.doc_id}-${e.span}`}>
                    <blockquote>“{e.span}”</blockquote>
                    <div className="quote-meta">
                      <span className="mono">{e.source}</span>
                      <span className="mono">{e.doc_id}</span>
                      {e.url && (
                        <a href={e.url} target="_blank" rel="noreferrer">
                          source
                        </a>
                      )}
                    </div>
                    <p className="fulltext">{e.text}</p>
                  </li>
                ))}
              </ul>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
