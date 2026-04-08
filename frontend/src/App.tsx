/**
 * AutoReasoner - Simple Code Comprehension Assistant
 * 
 * Main React component that implements the 4-step flow:
 * Step 1: User inputs GitHub URL
 * Step 4: Display analysis results
 */

import React, { useState } from "react";
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { analyzeRepo, getAnalysis, AnalysisResult, FileSummary } from "./api";
import { AuthProvider, useAuth } from './AuthContext';
import LoginPage from './LoginPage';
import RegisterPage from './RegisterPage';
import { FileTree } from './FileTree';
import { FileExplanationPage } from './FileExplanationPage';
import "./App.css";
import AnimatedBackground from './components/AnimatedBackground';

function RequireAuth({ children }: { children: JSX.Element }) {
  const { user, loading } = useAuth() as any;

  if (loading) {
    return <div style={{ color: 'white', textAlign: 'center', marginTop: '20%' }}>Loading...</div>;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

function Dashboard() {
  const [repoUrl, setRepoUrl] = useState("");
  const [forceRefresh, setForceRefresh] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const { logout, user, token } = useAuth() as any;

  /**
   * Step 1: Handle user clicking the Analyze button
   */
  const handleAnalyze = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a GitHub repository URL");
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      // If not forcing refresh, check cache first
      if (!forceRefresh) {
        const cached = await getAnalysis(repoUrl.trim(), token);
        if (cached) {
          setAnalysis(cached);
          setLoading(false);
          return;
        }
      }

      // Perform analysis (passing forceRefresh flag)
      const result = await analyzeRepo(repoUrl.trim(), token, forceRefresh);
      setAnalysis(result);
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            <h1>AutoReasoner</h1>
            <p>Intelligent Code Comprehension Assistant</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '14px' }}>Hi, {user?.username}</span>
            <button onClick={logout} style={{ padding: '8px 16px', borderRadius: '4px', border: 'none', background: 'rgba(255,255,255,0.2)', color: 'white', cursor: 'pointer' }}>Logout</button>
          </div>
        </div>
      </header>

      <main className="App-main">
        {/* Step 1: User input */}
        <div className="input-section">
          <input
            type="text"
            placeholder="Enter GitHub repository URL (e.g., https://github.com/facebook/react)"
            value={repoUrl}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setRepoUrl(e.target.value)}
            className="url-input"
            disabled={loading}
          />
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '8px', color: '#666' }}>
            <input
              type="checkbox"
              id="force-refresh"
              checked={forceRefresh}
              onChange={(e) => setForceRefresh(e.target.checked)}
              disabled={loading}
            />
            <label htmlFor="force-refresh" style={{ fontSize: '0.9rem', cursor: 'pointer' }}>Force Re-analysis (ignore cache)</label>
          </div>
          <button
            onClick={handleAnalyze}
            disabled={loading || !repoUrl.trim()}
            className="analyze-button"
            style={{ marginTop: '15px' }}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {/* Error display */}
        {error && <div className="error-message">{error}</div>}

        {/* Step 4: Display results */}
        {analysis && (
          <div className="results-section">
            <h2>Repository Summary</h2>
            <p className="repo-summary">{analysis.repo_summary}</p>

            <FileTree files={analysis.file_summaries} repoUrl={repoUrl.trim()} token={token} />

            <h2>File Summaries</h2>
            <table className="file-table">
              <thead>
                <tr>
                  <th>File Path</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {analysis.file_summaries.map((file: FileSummary, index: number) => (
                  <tr key={index}>
                    <td className="file-path">{file.path}</td>
                    <td className="file-summary">{file.summary}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}


function App() {
  return (
    <AuthProvider>
      <Router>
        <AnimatedBackground />
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={
            <RequireAuth>
              <Dashboard />
            </RequireAuth>
          } />
          <Route path="/explanation" element={
            <RequireAuth>
              <FileExplanationPage />
            </RequireAuth>
          } />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;

