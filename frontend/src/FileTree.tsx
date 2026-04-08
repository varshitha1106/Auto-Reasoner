import React, { useState } from 'react';
// Removed unused FileExplanation import
// Removed unused FileExplanation import

// Types
interface FileNode {
    name: string;
    path: string;
    type: 'file' | 'folder';
    children?: FileNode[];
    summary?: string;
    repoUrl?: string;
    token?: string;
}

const FileTreeItem = ({ node, repoUrl, token }: { node: FileNode, repoUrl: string, token: string }) => {
    const [expanded, setExpanded] = useState(false);
    // const navigate = useNavigate(); // Not needed for window.open

    const handleFileClick = () => {
        if (node.type === 'file') {
            // Open explanation in new tab
            const encodedRepo = encodeURIComponent(repoUrl);
            const encodedFile = encodeURIComponent(node.path);
            const url = `/#/explanation?repo=${encodedRepo}&file=${encodedFile}`;
            window.open(url, '_blank');
        } else {
            // Toggle folder
            setExpanded(!expanded);
        }
    };

    return (
        <div>
            <div
                onClick={handleFileClick}
                style={{
                    paddingLeft: '20px',
                    margin: '5px 0',
                    display: 'flex',
                    alignItems: 'center',
                    cursor: 'pointer',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    transition: 'background-color 0.2s',
                    backgroundColor: 'transparent'
                }}
                onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#f0f4f8';
                }}
                onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                }}
                title={node.type === 'file' ? "Click to view explanation" : "Click to expand/collapse"}
            >
                <span style={{ marginRight: '8px', fontSize: '0.9em' }}>
                    {node.type === 'folder' ? (expanded ? '📂' : '📁') : '📄'}
                </span>
                <span style={{
                    color: node.type === 'folder' ? '#333' : '#0366d6',
                    fontWeight: node.type === 'folder' ? 'bold' : 'normal'
                }}>
                    {node.name}
                </span>
            </div>

            {node.type === 'folder' && expanded && node.children && (
                <div style={{ marginLeft: '20px', borderLeft: '1px solid #eee' }}>
                    {node.children.map((child, index) => (
                        <FileTreeItem key={index} node={child} repoUrl={repoUrl} token={token} />
                    ))}
                </div>
            )}
        </div>
    );
};

export const buildFileTree = (files: { path: string; summary: string }[]): FileNodeLike[] => {
    const root: FileNodeLike[] = [];

    files.forEach(file => {
        const parts = file.path.split('/');
        let currentLevel = root;

        parts.forEach((part, index) => {
            const isFile = index === parts.length - 1;
            const existingNode = currentLevel.find(n => n.name === part);

            if (existingNode) {
                // If it's a folder, traverse into it
                if (!existingNode.children) {
                    existingNode.children = [];
                }
                currentLevel = existingNode.children;
            } else {
                const newNode: FileNodeLike = {
                    name: part,
                    path: isFile ? file.path : parts.slice(0, index + 1).join('/'),
                    type: isFile ? 'file' : 'folder',
                    summary: isFile ? file.summary : undefined,
                    children: isFile ? undefined : []
                };
                currentLevel.push(newNode);
                if (!isFile && newNode.children) {
                    currentLevel = newNode.children;
                }
            }
        });
    });

    return root;
};

// Simplified type for internal logic, exported strict type above for usage
interface FileNodeLike {
    name: string;
    path: string;
    type: 'file' | 'folder';
    children?: FileNodeLike[];
    summary?: string;
}

export const FileTree = ({ files, repoUrl, token }: { files: { path: string; summary: string }[], repoUrl: string, token: string }) => {
    const tree = buildFileTree(files);

    return (
        <div style={{ marginTop: '20px', background: '#fafafa', padding: '20px', borderRadius: '8px', border: '1px solid #eee' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>Repository Files (Click to explain)</h3>
            {tree.map((node, index) => (
                <FileTreeItem key={index} node={node as FileNode} repoUrl={repoUrl} token={token} />
            ))}
        </div>
    );
};
