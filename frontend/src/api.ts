/**
 * Simple API client functions for communicating with the FastAPI backend.
 */

const API_BASE_URL = "http://localhost:8000";

export interface FileSummary {
  path: string;
  summary: string;
}

export interface AnalysisResult {
  repo_summary: string;
  file_summaries: FileSummary[];
}

/**
 * Analyze a GitHub repository.
 * This performs Steps 2 and 3: fetching code and analyzing it.
 */
export async function analyzeRepo(repoUrl: string, token: string, forceRefresh: boolean = false): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    },
    body: JSON.stringify({
      repo_url: repoUrl,
      force_refresh: forceRefresh
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to analyze repository");
  }

  return response.json();
}

/**
 * Get stored analysis for a repository URL.
 * This is used in Step 4 to retrieve cached results.
 */
export async function getAnalysis(repoUrl: string, token: string): Promise<AnalysisResult | null> {
  // Note: GET requests can't have body, so we pass query params. 
  // But we still need auth header.
  const response = await fetch(
    `${API_BASE_URL}/analysis?repo_url=${encodeURIComponent(repoUrl)}`, {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${token}`
    }
  }
  );

  if (!response.ok) {
    if (response.status === 404) {
      return null; // Return null if not found
    }
    const error = await response.json();
    throw new Error(error.detail || "Failed to get analysis");
  }

  return response.json();
}


/**
 * Get (or generate) a detailed line-by-line explanation for a specific file.
 * Returns code content along with explanations.
 */
export interface FileExplanation {
  path: string;
  code: string;
  code_lines: string[];
  explanation: string;
  block_explanations?: Array<{
    start_line: number;
    end_line: number;
    code_block: string;
    explanation: string;
  }>;
  dependencies?: {
    file_dependencies: Array<{
      from: string;
      to: string;
      type: string;
    }>;
    function_dependencies: Array<{
      from: string;
      to: string;
      type: string;
    }>;
  };
}

export async function getFileExplanation(repoUrl: string, path: string, token: string): Promise<FileExplanation> {
  const response = await fetch(
    `${API_BASE_URL}/file-explanation?repo_url=${encodeURIComponent(repoUrl)}&path=${encodeURIComponent(path)}`, {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${token}`
    }
  }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to get file explanation");
  }

  return response.json();
}
