import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./Layout";
import ConfigJobPage from "./pages/ConfigJob";
import MonitoringPage from "./pages/Monitoring";
import ResultsPage from "./pages/Results";
import InsightsPage from "./pages/Insights";
import PromptOptPage from "./pages/PromptOpt";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ConfigJobPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/insights" element={<InsightsPage />} />
          <Route path="/prompt-opt" element={<PromptOptPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
