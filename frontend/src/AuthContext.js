import React, { createContext, useState, useEffect, useContext } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if token exists and validate
        const validateToken = async () => {
            if (token) {
                try {
                    const response = await fetch('http://localhost:8000/users/me', {
                        headers: { Authorization: `Bearer ${token}` }
                    });
                    if (response.ok) {
                        const userData = await response.json();
                        setUser(userData);
                    } else {
                        logout();
                    }
                } catch (error) {
                    logout();
                }
            }
            setLoading(false);
        };
        validateToken();
    }, [token]);

    const login = async (username, password) => {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch('http://localhost:8000/token', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Login failed');
        }

        const data = await response.json();
        const accessToken = data.access_token;
        localStorage.setItem('token', accessToken);
        setToken(accessToken);

        // Fetch user profile immediately
        try {
            const userResponse = await fetch('http://localhost:8000/users/me', {
                headers: { Authorization: `Bearer ${accessToken}` }
            });
            if (userResponse.ok) {
                const userData = await userResponse.json();
                setUser(userData);
            }
        } catch (error) {
            console.error("Failed to fetch user profile", error);
        }

        return true;
    };

    const register = async (username, password) => {
        const response = await fetch('http://localhost:8000/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }
        return true;
    };

    const logout = () => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, token, login, logout, register, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
