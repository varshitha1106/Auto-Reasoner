# AutoReasoner - Simple Code Comprehension Assistant

A beginner-friendly tool that analyzes GitHub repositories using AST parsing and LLM-based code summarization.

## Features

- **4-Step Simple Flow:**
  1. User pastes GitHub repository URL
  2. Backend fetches code from GitHub API
  3. Code is analyzed with TreeSitter (AST) and Hugging Face CodeT5 (LLM)
  4. Results displayed in a simple React interface

- **Secure Authentication:**
  - User registration and login
  - JWT-based session management
  - Protected dashboard routes

- **Visualizations:**
  - Interactive File Tree explorer
  - Detailed file-by-file code summaries

- **Smart Caching:** Results are stored in SQLite for instant re-analysis

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Set up GitHub token:**
   - Create a `.env` file in the project root
   - Add your GitHub token: `GITHUB_TOKEN=your_token_here`

## How to Run the Application

### Option 1: One-Click Start (Recommended)
1. **Double-click** the `start_app.bat` file in the project root.
2. Two command windows will open (one for the backend, one for the frontend).
3. The browser will automatically open at `http://localhost:3000`.

### Option 2: Manual Start
If the batch file doesn't work, you can start the servers manually in two separate terminal windows:

**Terminal 1 (Backend):**
```bash
python -m uvicorn backend.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm start
```

## Usage

1. **Register/Login**: Create an account to access the dashboard.
2. **Analyze**: Paste a GitHub URL (e.g., `https://github.com/facebook/react`).
3. **Explore**:
   - View repository summary.
   - Navigate the **File Tree** to see the structure.
   - Read summaries for individual files.

## Project Structure

```
.
├── backend/
│   ├── main.py           # FastAPI app (Auth, Analyze endpoints)
│   ├── github_client.py  # GitHub API client
│   ├── analyzer.py       # AST parsing + LLM summarization
│   └── db.py             # SQLite database (Users, Cache)
├── frontend/
│   ├── src/
│   │   ├── App.tsx       # Main Layout & Routing
│   │   ├── FileTree.tsx  # Recursive File Tree Component
│   │   ├── AuthContext.js# User Session Management
│   │   ├── LoginPage.js  # Login View
│   │   ├── RegisterPage.js # Register View
│   │   └── api.ts        # API client
│   └── package.json
├── requirements.txt      # Python dependencies
├── start_app.bat         # OPTIMIZED: Single startup script
└── README.md
```

## Technologies Used

- **Backend:** Python 3.10, FastAPI, GitHub REST API, Python-Jose (JWT), Passlib (Bcrypt)
- **Code Analysis:** TreeSitter (AST), Hugging Face CodeT5 (LLM)
- **Database:** SQLite
- **Frontend:** React, TypeScript, React Router
