
import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { getFileExplanation, FileExplanation } from './api';
import { useAuth } from './AuthContext';

export const FileExplanationPage = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { token } = useAuth() as any;

  const repoUrl = searchParams.get('repo');
  const filePath = searchParams.get('file');

  const [data, setData] = useState<FileExplanation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!repoUrl || !filePath || !token) {
      setError("Missing parameters");
      setLoading(false);
      return;
    }

    const fetchData = async () => {
      try {
        const result = await getFileExplanation(repoUrl, filePath, token);
        setData(result);
      } catch (err: any) {
        setError(err.message || "Failed to load explanation");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [repoUrl, filePath, token]);

  // Function to parse and style the explanation text
  const renderStyledExplanation = (text: string) => {
    if (!text) return null;

    return text.split('\n').map((line, index) => {
      // Check for --- HEADINGS ---
      const headingMatch = line.match(/^--- (.+) ---$/);
      if (headingMatch) {
        return (
          <div key={index} style={{
            fontWeight: 'bold',
            fontStyle: 'italic',
            fontSize: '1.2em',
            marginTop: '20px',
            marginBottom: '10px',
            color: '#2c5282',
            borderBottom: '1px solid #e2e8f0',
            paddingBottom: '5px'
          }}>
            {headingMatch[1]}
          </div>
        );
      }

      // Check for Block headers like "Block 1: ..." or "Line 1:"
      if (line.match(/^(Block \d+:|Line \d+:)/)) {
        return (
          <div key={index} style={{
            fontWeight: 'bold',
            marginTop: '10px',
            color: '#4a5568'
          }}>
            {line}
          </div>
        );
      }

      return (
        <div key={index} style={{ minHeight: '1.5em' }}>
          {line}
        </div>
      );
    });
  };

  if (loading) return <div style={{ padding: '40px', textAlign: 'center' }}>Loading explanation...</div>;

  if (error) return (
    <div style={{ padding: '20px', color: 'red' }}>
      Error: {error}
      <button onClick={() => navigate(-1)} style={{ marginLeft: '10px' }}>Go Back</button>
    </div>
  );

  if (!data) return <div>No data found</div>;

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>

      <h1 style={{ fontSize: '1.5em', marginBottom: '20px', color: '#2d3748' }}>
        Explanation: {data.path}
      </h1>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Left Column: Code */}
        <div style={{
          background: '#282c34',
          color: '#abb2bf',
          padding: '15px',
          borderRadius: '8px',
          overflow: 'auto',
          maxHeight: '80vh'
        }}>
          <h3 style={{ color: '#e0e0e0', marginTop: 0 }}>Source Code</h3>
          <pre style={{ margin: 0, fontFamily: 'Consolas, monospace' }}>
            {data.code_lines.map((line, i) => (
              <div key={i} style={{ display: 'flex' }}>
                <span style={{
                  width: '30px',
                  display: 'inline-block',
                  color: '#5c6370',
                  userSelect: 'none',
                  textAlign: 'right',
                  marginRight: '10px'
                }}>{i + 1}</span>
                <span>{line}</span>
              </div>
            ))}
          </pre>
        </div>

        {/* Right Column: Explanation */}
        <div style={{
          background: 'white',
          padding: '20px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          overflow: 'auto',
          maxHeight: '80vh',
          lineHeight: '1.6'
        }}>
          {renderStyledExplanation(data.explanation)}

          {/* Dependencies Section */}
          {data.dependencies && (
            <div style={{ marginTop: '30px', borderTop: '2px solid #e2e8f0', paddingTop: '20px' }}>
              <h3 style={{ color: '#2d3748', borderBottom: '1px solid #cbd5e0', paddingBottom: '10px' }}>Dependencies</h3>

              {/* File Dependencies */}
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: '#4a5568', margin: '10px 0' }}>File-Level Dependencies</h4>
                {data.dependencies.file_dependencies && data.dependencies.file_dependencies.length > 0 ? (
                  <ul style={{ listStyleType: 'none', padding: 0 }}>
                    {data.dependencies.file_dependencies.map((dep, idx) => (
                      <li key={idx} style={{
                        background: '#f7fafc',
                        padding: '10px',
                        marginBottom: '8px',
                        borderRadius: '4px',
                        borderLeft: '4px solid #4299e1',
                        fontSize: '0.9em'
                      }}>
                        <span style={{ fontWeight: 'bold' }}>{dep.from}</span>
                        <span style={{ margin: '0 8px', color: '#718096' }}>→</span>
                        <span style={{ color: '#2b6cb0' }}>{dep.to}</span>
                        <span style={{ float: 'right', fontSize: '0.8em', color: '#a0aec0', textTransform: 'uppercase' }}>{dep.type}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ fontStyle: 'italic', color: '#718096' }}>No file dependencies found.</p>
                )}
              </div>

              {/* Function Dependencies */}
              <div>
                <h4 style={{ color: '#4a5568', margin: '10px 0' }}>Function-Level Call Flow</h4>
                {data.dependencies.function_dependencies && data.dependencies.function_dependencies.length > 0 ? (
                  <ul style={{ listStyleType: 'none', padding: 0 }}>
                    {data.dependencies.function_dependencies.map((dep, idx) => (
                      <li key={idx} style={{
                        background: '#fffaf0',
                        padding: '10px',
                        marginBottom: '8px',
                        borderRadius: '4px',
                        borderLeft: '4px solid #ed8936',
                        fontSize: '0.9em'
                      }}>
                        <span style={{ fontWeight: 'bold' }}>{dep.from}</span>
                        <span style={{ margin: '0 8px', color: '#718096' }}>→</span>
                        <span style={{ color: '#c05621' }}>{dep.to}</span>
                        <span style={{ float: 'right', fontSize: '0.8em', color: '#a0aec0', textTransform: 'uppercase' }}>{dep.type}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p style={{ fontStyle: 'italic', color: '#718096' }}>No function dependencies found.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
