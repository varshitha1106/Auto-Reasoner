import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from './AuthContext';
import './App.css'; // Assuming we can piggyback on global styles or I'll inject styles

const LoginPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        try {
            await login(username, password);
            navigate('/');
        } catch (err) {
            setError(err.message || 'Failed to login');
        }
    };

    const containerStyle = {
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    };

    const cardStyle = {
        background: 'rgba(255, 255, 255, 0.25)',
        backdropFilter: 'blur(10px)',
        borderRadius: '16px',
        padding: '3rem',
        width: '400px',
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        border: '1px solid rgba(255, 255, 255, 0.18)',
        color: 'white',
    };

    const inputStyle = {
        width: '100%',
        padding: '12px 20px',
        margin: '8px 0',
        display: 'inline-block',
        border: '1px solid rgba(255,255,255,0.3)',
        borderRadius: '8px',
        boxSizing: 'border-box',
        background: 'rgba(255,255,255,0.1)',
        color: 'white',
        outline: 'none',
        fontSize: '16px',
    };

    const buttonStyle = {
        width: '100%',
        backgroundColor: '#fff',
        color: '#764ba2',
        padding: '14px 20px',
        margin: '24px 0 12px 0',
        border: 'none',
        borderRadius: '8px',
        cursor: 'pointer',
        fontSize: '16px',
        fontWeight: 'bold',
        transition: 'transform 0.2s',
    };

    return (
        <div style={containerStyle}>
            <div style={cardStyle}>
                <h2 style={{ textAlign: 'center', marginBottom: '2rem', fontSize: '2rem' }}>Welcome Back</h2>
                {error && <div style={{ color: '#ffaaaa', marginBottom: '1rem', textAlign: 'center' }}>{error}</div>}
                <form onSubmit={handleSubmit}>
                    <div style={{ marginBottom: '1rem' }}>
                        <label>Username</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            style={inputStyle}
                            placeholder="Enter username"
                            required
                        />
                    </div>
                    <div style={{ marginBottom: '1rem' }}>
                        <label>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            style={inputStyle}
                            placeholder="Enter password"
                            required
                        />
                    </div>
                    <button
                        type="submit"
                        style={buttonStyle}
                        onMouseOver={(e) => e.target.style.transform = 'scale(1.02)'}
                        onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
                    >
                        Sign In
                    </button>
                    <div style={{ textAlign: 'center', marginTop: '1rem' }}>
                        Don't have an account? <Link to="/register" style={{ color: 'white', fontWeight: 'bold' }}>Register</Link>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
