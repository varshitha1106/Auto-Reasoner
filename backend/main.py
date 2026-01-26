
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.db import init_db, get_analysis, save_analysis, create_user, get_user
from backend.github_client import fetch_repository_code
from backend.analyzer import parse_files_with_treesitter, summarize_repository

# Initialize FastAPI app
app = FastAPI(title="AutoReasoner API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth Configurations
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()


# Request/Response models
class AnalyzeRequest(BaseModel):
    repo_url: str
    force_refresh: bool = False


class AnalysisResponse(BaseModel):
    repo_summary: str
    file_summaries: list


class RegisterRequest(BaseModel):
    username: str
    password: str


@app.post("/register")
async def register(request: RegisterRequest):
    if get_user(request.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(request.password)
    create_user(request.username, hashed_password)
    return {"message": "User created successfully"}


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repository(request: AnalyzeRequest, current_user: dict = Depends(get_current_user)):

    repo_url = request.repo_url.strip()
    
    # Check if we already have this analysis
    if not request.force_refresh:
        cached = get_analysis(repo_url)
        if cached:
            return AnalysisResponse(
                repo_summary=cached["repo_summary"],
                file_summaries=cached["file_summaries"]
            )
    
    try:
        # Step 2: Fetch repository code from GitHub API
        print(f"Fetching code from {repo_url}...")
        files = fetch_repository_code(repo_url)
        
        if not files:
            raise HTTPException(status_code=400, detail="No code files found in repository")
        
        # Step 3: Analyze code with TreeSitter and LLM
        print(f"Parsing {len(files)} files with TreeSitter...")
        parsed_data = parse_files_with_treesitter(files)
        
        print("Generating summaries with LLM...")
        analysis = summarize_repository(parsed_data)
        
        # Save to database
        save_analysis(
            repo_url,
            analysis["repo_summary"],
            analysis["file_summaries"]
        )
        
        return AnalysisResponse(
            repo_summary=analysis["repo_summary"],
            file_summaries=analysis["file_summaries"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing repository: {str(e)}")



@app.get("/analysis")
async def get_stored_analysis(repo_url: str):
   
    cached = get_analysis(repo_url)
    if cached:
        return AnalysisResponse(
            repo_summary=cached["repo_summary"],
            file_summaries=cached["file_summaries"]
        )
    else:
        raise HTTPException(status_code=404, detail="Analysis not found")


@app.get("/file-explanation")
async def get_file_explanation_endpoint(repo_url: str, path: str, current_user: dict = Depends(get_current_user)):
    """
    Get or generate a detailed line-by-line explanation for a specific file.
    Returns both code content and explanations.
    """
    from backend.db import get_file_explanation, save_file_explanation
    from backend.analyzer import generate_single_file_explanation
    from backend.github_client import fetch_single_file
    import json

    repo_url = repo_url.strip()

    # Check cache first - try to get cached explanation
    cached_explanation = get_file_explanation(repo_url, path)
    
    # Not found, generate it
    try:
        # Load code
        print(f"Fetching file content for {path}...")
        code = fetch_single_file(repo_url, path)
        if not code:
             raise HTTPException(status_code=404, detail="File content could not be fetched")
        
        # If we have cached explanation, try to parse it as structured data
        # Otherwise generate new one
        if cached_explanation:
            try:
                # Try to parse as JSON (new format)
                cached_data = json.loads(cached_explanation)
                if isinstance(cached_data, dict) and "code" in cached_data:
                    # Check if dependencies exist (new feature)
                    if "dependencies" in cached_data:
                        return {
                            "path": path,
                            "code": cached_data["code"],
                            "code_lines": cached_data["code_lines"],
                            "explanation": cached_data["explanation"],
                            "block_explanations": cached_data.get("block_explanations", []),
                            "dependencies": cached_data["dependencies"]
                        }
                    # If dependencies missing, fall through to regenerate
            except (json.JSONDecodeError, KeyError):
                # Old format - just explanation string, generate new structured format
                pass
        
        # Generate new structured explanation
        print(f"Generating line-by-line explanation for {path}...")
        result = generate_single_file_explanation(code, path)
        
        # Save to DB as JSON string
        save_file_explanation(repo_url, path, json.dumps(result))
        
        return {
            "path": path,
            "code": result["code"],
            "code_lines": result["code_lines"],
            "explanation": result["explanation"],
            "block_explanations": result.get("block_explanations", []),
            "dependencies": result.get("dependencies", {})
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error generating explanation for {path}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("error_log.txt", "a") as f:
            f.write(f"[{datetime.utcnow()}] {error_msg}\n")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "AutoReasoner API is running"}


