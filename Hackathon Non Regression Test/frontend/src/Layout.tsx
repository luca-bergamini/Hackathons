import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", icon: "⚙️", label: "Config Job" },
  { to: "/monitoring", icon: "📡", label: "Monitoring" },
  { to: "/results", icon: "📊", label: "Results" },
  { to: "/insights", icon: "🔍", label: "Insight & Compare" },
  { to: "/prompt-opt", icon: "✨", label: "Prompt Optimization" },
];

export default function Layout() {
  return (
    <div className="app-layout">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="topbar-brand">
            <img src="/logo.png" alt="Logo" className="brand-logo" />
            <h1>NRT Pipeline</h1>
          </div>

          <nav className="topbar-nav">
            {NAV_ITEMS.map(({ to, icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `nav-link ${isActive ? "active" : ""}`
                }
              >
                <span className="nav-emoji">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
