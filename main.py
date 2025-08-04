#!/usr/bin/env python3
"""
AI Agent System - Main FastAPI Application
Comprehensive agent management with RAG capabilities and outbound calling
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
import requests
import PyPDF2
import docx
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks, Body, Depends, FastAPI, HTTPException, Request
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pinecone import Pinecone

# Load environment variables
load_dotenv(dotenv_path=".env")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-agent-system")

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
http_bearer = HTTPBearer()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ai_agent_demo")

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client[MONGO_DB_NAME]
    agents_collection = db["agents"]
    documents_collection = db["documents"]
    calls_collection = db["calls"]
    users_collection = db["users"]
    mongo_client.server_info()
    logger.info("Successfully connected to MongoDB")
except ServerSelectionTimeoutError:
    logger.error("Failed to connect to MongoDB. Check your credentials/URI!")
    db = None
    agents_collection = None
    documents_collection = None
    calls_collection = None
    users_collection = None

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "luminous-pine")

if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY not found - RAG features will be disabled")
    pc = None
    index = None
else:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        index_stats = index.describe_index_stats()
        logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        logger.info(f"Index stats: {index_stats}")
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {str(e)}")
        pc = None
        index = None

# FastAPI app
app = FastAPI(
    title="AI Agent Creation API", 
    description="Create AI agents with RAG capabilities and outbound calling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        raise

# Health Check Endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mongodb": "connected" if db is not None else "disconnected",
        "pinecone": "connected" if index is not None else "disconnected"
    }

# === Pydantic Models ===

class UserBase(BaseModel):
    email: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str
    disabled: Optional[bool] = False

class UserResponse(UserBase):
    id: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the AI agent")
    language: str = Field(default="English", description="Primary language for the agent")
    model: str = Field(default="gpt-4", description="OpenAI model to use")
    persona: str = Field(..., min_length=1, description="Personality and behavior description")
    rag_docs: Optional[List[str]] = Field(default=[], description="List of document URLs or content for RAG")
    instructions: Optional[str] = Field(default=None, description="Additional instructions for the agent")
    description: Optional[str] = Field(default=None, description="Description of the agent's purpose")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity level")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens per response")
    plan: Optional[str] = None
    default_voice: Optional[str] = "alloy"  # Set default voice
    webhook_transcript: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None
    custom_prompts: Optional[Dict[str, Any]] = None
    webhook_custom_fields: Optional[str] = None
    default_tone: Optional[str] = "professional"  # Set default tone
    metadata: Optional[Dict[str, Any]] = None
    virtual_number: Optional[str] = None

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    language: str
    model: str
    persona: str
    instructions: Optional[str]
    description: Optional[str]
    temperature: float
    max_tokens: int
    status: str
    rag_status: str
    rag_docs_count: int
    vector_namespace: str
    created_at: datetime
    updated_at: datetime
    embedding_model: str
    total_chunks: int
    processing_time_seconds: Optional[float]
    plan: Optional[str] = None
    default_voice: Optional[str] = "alloy"
    webhook_transcript: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None
    custom_prompts: Optional[Dict[str, Any]] = None
    webhook_custom_fields: Optional[str] = None
    default_tone: Optional[str] = "professional"
    metadata: Optional[Dict[str, Any]] = None
    virtual_number: Optional[str] = None

class CallRequest(BaseModel):
    agent_id: str
    voice_actor: Optional[str] = None
    tone: Optional[str] = None
    prompt_vars: Optional[Dict[str, str]] = {}
    metadata: Optional[Dict[str, str]] = {}
    phone_number: str

class CallResponse(BaseModel):
    call_id: str
    status: str

# === Authentication Functions ===

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(email: str):
    if users_collection is None:
        return None
    user = users_collection.find_one({"email": email})
    if user:
        user["id"] = str(user.get("_id"))
        return user
    return None

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = get_user(email)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_db():
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    return db

# === Utility Functions ===

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        chunk = text[start:end]
        # Find sentence boundaries
        last_period = chunk.rfind('.')
        last_question = chunk.rfind('?')
        last_exclamation = chunk.rfind('!')
        sentence_end = max(last_period, last_question, last_exclamation)
        
        if sentence_end > start + chunk_size // 2:
            chunks.append(text[start:start + sentence_end + 1])
            start = start + sentence_end + 1 - overlap
        else:
            chunks.append(chunk)
            start = end - overlap
    
    return chunks

def fetch_url_content(url: str) -> dict:
    """Fetch and extract content from URLs"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        content_type = resp.headers.get("content-type", "").lower()
        
        # Handle PDF files
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            temp_filename = f"temp_{uuid.uuid4().hex[:8]}.pdf"
            try:
                with open(temp_filename, "wb") as f:
                    f.write(resp.content)
                with open(temp_filename, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                os.remove(temp_filename)
                return {"title": url, "content": clean_text(text), "url": url, "status": "success"}
            except Exception as e:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise e
        
        # Handle plain text files
        if "text/plain" in content_type or url.lower().endswith(".txt"):
            return {"title": url, "content": clean_text(resp.text), "url": url, "status": "success"}
        
        # Handle HTML content
        html = resp.text
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else url
        
        # Remove script and style tags
        html = re.sub(r"<(script|style).*?>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        
        return {"title": title, "content": clean_text(text), "url": url, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return {"title": url, "content": "", "url": url, "status": "failed", "error": str(e)}

def extract_text_from_source(source: str, i: int) -> tuple[str, str]:
    """Extract text from various sources (URLs, files, or direct text)"""
    try:
        if source.startswith(('http://', 'https://')):
            doc_data = fetch_url_content(source)
            return doc_data["content"], doc_data["title"]
        elif source.lower().endswith(".pdf") and os.path.exists(source):
            with open(source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return clean_text(text), os.path.basename(source)
        elif source.lower().endswith(".txt") and os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                return clean_text(f.read()), os.path.basename(source)
        elif source.lower().endswith(".docx") and os.path.exists(source):
            doc = docx.Document(source)
            text = "\n".join([para.text for para in doc.paragraphs])
            return clean_text(text), os.path.basename(source)
        else:
            # Treat as direct text content
            return clean_text(source), f"Document {i+1}"
    except Exception as e:
        logger.error(f"Failed to extract from {source}: {str(e)}")
        return "", f"Document {i+1}"

def generate_doc_id(content: str, source: str) -> str:
    """Generate unique document ID"""
    content_hash = hashlib.md5(f"{source}{content}".encode()).hexdigest()[:8]
    return f"doc_{content_hash}"

def generate_embedding(text: str, model: str = "text-embedding-3-small"):
    """Generate embeddings using OpenAI API"""
    try:
        import openai
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is required for embeddings")
            return None
            
        client = openai.OpenAI(api_key=openai_api_key)
        
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def generate_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100):
    """Generate embeddings for multiple texts in batches"""
    try:
        import openai
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable is required for embeddings")
            return None
            
        client = openai.OpenAI(api_key=openai_api_key)
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                all_embeddings.extend([None] * len(batch))
        
        return all_embeddings
    except Exception as e:
        logger.error(f"Error in batch embedding generation: {str(e)}")
        return None

async def query_agent_knowledge(agent_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """Query the agent's knowledge base from Pinecone"""
    if index is None:
        logger.warning("Pinecone index not available")
        return []
        
    try:
        namespace = f"agent_{agent_id}"
        
        logger.info(f"Querying Pinecone namespace: {namespace} for query: {query}")
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        logger.info(f"Pinecone query returned {len(results.matches)} matches")
        
        # Extract relevant context
        contexts = []
        if results.matches:
            for match in results.matches:
                logger.info(f"Match score: {match.score}, metadata: {match.metadata}")
                if match.score > 0.3:  # Relevance threshold
                    contexts.append({
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", ""),
                        "title": match.metadata.get("title", ""),
                        "score": match.score
                    })
        
        logger.info(f"Filtered contexts: {len(contexts)} items above threshold")
        return contexts
    except Exception as e:
        logger.error(f"Error querying agent knowledge: {e}")
        return []

async def process_rag_documents(agent_id: str, rag_docs: list):
    """Process RAG documents and store in vector database"""
    start_time = time.time()
    namespace = f"agent_{agent_id}"
    total_chunks = 0
    processed_docs = 0
    
    try:
        # Update agent status to processing
        if agents_collection is not None:
            agents_collection.update_one(
                {"agent_id": agent_id},
                {"$set": {"rag_status": "processing", "updated_at": datetime.utcnow()}}
            )
        
        vectors_to_upsert = []
        
        for i, doc_source in enumerate(rag_docs):
            logger.info(f"Processing document {i+1}/{len(rag_docs)}: {doc_source}")
            content, title = extract_text_from_source(doc_source, i)
            
            if not content:
                logger.warning(f"No content extracted from {doc_source}")
                continue
            
            doc_id = generate_doc_id(content, doc_source)
            chunks = chunk_text(content)
            
            for j, chunk in enumerate(chunks):
                vector_id = f"{agent_id}_{doc_id}_chunk_{j}"
                
                metadata = {
                    "agent_id": agent_id,
                    "doc_id": doc_id,
                    "source": doc_source,
                    "title": title,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "content_length": len(chunk),
                    "created_at": datetime.utcnow().isoformat(),
                    "text": chunk
                }
                
                vector_dict = {
                    "id": vector_id,
                    "metadata": metadata
                }
                
                vectors_to_upsert.append(vector_dict)
            
            # Store document metadata in MongoDB
            if documents_collection is not None:
                documents_collection.update_one(
                    {"doc_id": doc_id},
                    {"$set": {
                        "doc_id": doc_id,
                        "agent_id": agent_id,
                        "source": doc_source,
                        "title": title,
                        "content": content,
                        "chunks_count": len(chunks),
                        "processed_at": datetime.utcnow(),
                        "status": "processed"
                    }},
                    upsert=True
                )
            
            total_chunks += len(chunks)
            processed_docs += 1
            logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
        
        # Process embeddings and upsert to Pinecone
        if vectors_to_upsert:
            logger.info(f"Generating embeddings for {len(vectors_to_upsert)} chunks...")
            
            texts = [vector["metadata"]["text"] for vector in vectors_to_upsert]
            embeddings = generate_embeddings_batch(texts, batch_size=100)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                if agents_collection is not None:
                    agents_collection.update_one(
                        {"agent_id": agent_id},
                        {"$set": {"rag_status": "failed", "updated_at": datetime.utcnow()}}
                    )
                return
            
            # Add embeddings to vectors
            vectors_with_embeddings = []
            for i, (vector, embedding) in enumerate(zip(vectors_to_upsert, embeddings)):
                if embedding is not None:
                    vector_with_embedding = vector.copy()
                    vector_with_embedding["values"] = embedding
                    vectors_with_embeddings.append(vector_with_embedding)
                else:
                    logger.warning(f"Skipping vector {i} due to embedding generation failure")
            
            logger.info(f"Successfully generated {len(vectors_with_embeddings)} embeddings")
            
            # Perform batch upsert
            batch_size = 100
            successful_batches = 0
            
            for i in range(0, len(vectors_with_embeddings), batch_size):
                batch = vectors_with_embeddings[i:i + batch_size]
                try:
                    result = index.upsert(
                        vectors=batch,
                        namespace=namespace
                    )
                    logger.info(f"Upserted batch {i//batch_size + 1}: {result}")
                    successful_batches += 1
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                    continue
            
            # Update agent status
            processing_time = time.time() - start_time
            if successful_batches > 0 and agents_collection is not None:
                agents_collection.update_one(
                    {"agent_id": agent_id},
                    {"$set": {
                        "rag_status": "completed",
                        "rag_docs_count": processed_docs,
                        "total_chunks": total_chunks,
                        "processing_time_seconds": processing_time,
                        "successful_batches": successful_batches,
                        "updated_at": datetime.utcnow()
                    }}
                )
                logger.info(f"Successfully processed {processed_docs} documents with {total_chunks} chunks for agent {agent_id}")
            else:
                if agents_collection is not None:
                    agents_collection.update_one(
                        {"agent_id": agent_id},
                        {"$set": {"rag_status": "failed", "updated_at": datetime.utcnow()}}
                    )
        else:
            logger.warning(f"No valid vectors to upsert for agent {agent_id}")
            if agents_collection is not None:
                agents_collection.update_one(
                    {"agent_id": agent_id},
                    {"$set": {"rag_status": "failed", "updated_at": datetime.utcnow()}}
                )
    
    except Exception as e:
        logger.error(f"Error processing RAG documents for agent {agent_id}: {str(e)}")
        if agents_collection is not None:
            agents_collection.update_one(
                {"agent_id": agent_id},
                {"$set": {"rag_status": "failed", "updated_at": datetime.utcnow()}}
            )

# === API Routes ===

# Authentication Routes
@app.post("/api/register", response_model=UserResponse)
def register(user: UserCreate):
    """Register a new user"""
    if users_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    user_dict.pop("password")
    
    result = users_collection.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    
    return UserResponse(id=user_dict["id"], email=user.email, full_name=user.full_name)

@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/logout")
def logout():
    """Logout (client-side token removal)"""
    return {"detail": "Logged out"}

@app.get("/api/me", response_model=UserResponse)
def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return UserResponse(
        id=current_user["id"], 
        email=current_user["email"], 
        full_name=current_user.get("full_name")
    )

# Agent Management Routes
@app.post("/api/agent", response_model=AgentResponse)
async def create_agent(
    payload: AgentCreateRequest, 
    background_tasks: BackgroundTasks,
    db = Depends(get_db)
):
    """Create a new AI agent with RAG capabilities"""
    try:
        agent_id = "agent_" + uuid.uuid4().hex[:12]
        namespace = f"agent_{agent_id}"
        
        # Ensure default values are set
        default_voice = payload.default_voice or "alloy"
        default_tone = payload.default_tone or "professional"
        
        agent_data = {
            "agent_id": agent_id,
            "name": payload.name,
            "language": payload.language,
            "model": payload.model,
            "persona": payload.persona,
            "rag_docs": payload.rag_docs or [],
            "instructions": payload.instructions,
            "description": payload.description,
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "status": "active",
            "rag_status": "none" if not payload.rag_docs else "pending",
            "rag_docs_count": 0,
            "total_chunks": 0,
            "vector_namespace": namespace,
            "embedding_model": "text-embedding-3-small",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "processing_time_seconds": None,
            "plan": payload.plan,
            "default_voice": default_voice,
            "webhook_transcript": payload.webhook_transcript,
            "custom_fields": payload.custom_fields,
            "custom_prompts": payload.custom_prompts,
            "webhook_custom_fields": payload.webhook_custom_fields,
            "default_tone": default_tone,
            "metadata": payload.metadata,
            "virtual_number": payload.virtual_number,
        }
        
        agents_collection.insert_one(agent_data)
        
        # Start RAG processing in background if documents provided
        if payload.rag_docs:
            background_tasks.add_task(process_rag_documents, agent_id, payload.rag_docs)
        
        response_data = AgentResponse(**agent_data)
        
        logger.info(f"Created agent {agent_id} with {len(payload.rag_docs)} RAG documents")
        return response_data
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.get("/api/agent/{agent_id}")
def get_agent(agent_id: str):
    """Get agent by ID"""
    if agents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    agent = agents_collection.find_one({"agent_id": agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent["id"] = agent.get("agent_id")
    agent.pop("_id", None)
    return agent

@app.get("/api/agents")
def list_agents(skip: int = 0, limit: int = 50):
    """List all agents with pagination"""
    if agents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    agents = list(agents_collection.find().skip(skip).limit(limit))
    for agent in agents:
        agent["id"] = str(agent.get("agent_id", agent.get("_id")))
        agent.pop("_id", None)
    
    return jsonable_encoder(agents)

@app.patch("/api/agent/{agent_id}", response_model=AgentResponse)
def patch_agent(agent_id: str, updates: dict = Body(...)):
    """Update agent configuration"""
    if agents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    updates["updated_at"] = datetime.utcnow()
    result = agents_collection.update_one({"agent_id": agent_id}, {"$set": updates})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_collection.find_one({"agent_id": agent_id})
    agent["id"] = agent.get("agent_id")
    agent.pop("_id", None)
    return agent

@app.delete("/api/agent/{agent_id}")
def delete_agent(agent_id: str):
    """Delete an AI agent by agent_id"""
    if agents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    result = agents_collection.delete_one({"agent_id": agent_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Optionally delete associated documents and vectors
    if documents_collection is not None:
        documents_collection.delete_many({"agent_id": agent_id})
    
    if index is not None:
        try:
            # Delete vectors from Pinecone namespace
            namespace = f"agent_{agent_id}"
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            logger.warning(f"Failed to delete Pinecone vectors for agent {agent_id}: {e}")
    
    return {"detail": f"Agent {agent_id} deleted"}

@app.get("/api/agent/{agent_id}/query")
async def query_agent(agent_id: str, query: str, top_k: int = 5):
    """Query an agent's knowledge base"""
    try:
        if index is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        namespace = f"agent_{agent_id}"
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Query with the generated embedding
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        return {
            "agent_id": agent_id,
            "query": query,
            "results": [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in (results.matches or [])
            ],
            "total_matches": len(results.matches) if results.matches else 0
        }
        
    except Exception as e:
        logger.error(f"Error querying agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query agent: {str(e)}")

# Call Management Routes
@app.post("/api/call", response_model=CallResponse)
async def call_agent(payload: CallRequest = Body(...)):
    """Initiate an outbound call using the specified agent with RAG capabilities"""
    if agents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    # Find the agent
    agent = agents_collection.find_one({"agent_id": payload.agent_id})
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {payload.agent_id} not found")
    
    # Only allow calls if RAG is ready (or not used)
    if agent.get("rag_docs") and agent.get("rag_status") != "completed":
        raise HTTPException(
            status_code=409, 
            detail=f"Agent RAG is not ready. Current status: {agent.get('rag_status')}"
        )
    
    # Create call record
    call_id = "call_" + uuid.uuid4().hex
    call_info = payload.model_dump()
    call_info['call_id'] = call_id
    call_info['status'] = 'initiated'
    call_info['created_at'] = datetime.utcnow()
    call_info['agent_id'] = payload.agent_id
    
    if calls_collection is not None:
        calls_collection.insert_one(call_info)
    
    # Prepare dial info with agent_id - ensure voice is never None
    voice_actor = payload.voice_actor or agent.get("default_voice") or "alloy"
    tone = payload.tone or agent.get("default_tone") or "professional"
    
    dial_info = {
        "agent_id": payload.agent_id,
        "phone_number": payload.phone_number,
        "transfer_to": (agent.get("custom_fields") or {}).get("transfer_to", None),
        "prompt_vars": payload.prompt_vars,
        "voice_actor": voice_actor,
        "tone": tone,
        "metadata": payload.metadata,
    }
    
    logger.info(f"[CALL] Using Pinecone namespace: agent_{payload.agent_id} for agent {payload.agent_id}")
    logger.info(f"[CALL] Voice setting: {voice_actor}, Tone: {tone}")
    
    # Dispatch the call using LiveKit CLI (simplified version)
    metadata_json = json.dumps(dial_info)
    try:
        # Don't use shlex.quote - it's adding extra quotes
        cmd = [
            "lk", "dispatch", "create",
            "--new-room",
            "--agent-name", "outbound-caller",
            "--metadata", metadata_json
        ]
        
        logger.info(f"Dispatching call with CLI command")
        logger.info(f"Metadata being sent: {metadata_json}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"LiveKit CLI dispatch failed: {result.stderr}")
            logger.error(f"CLI stdout: {result.stdout}")
            raise Exception(f"LiveKit CLI dispatch failed: {result.stderr}")
        
        logger.info(f"LiveKit CLI dispatch output: {result.stdout}")
        logger.info(f"Dispatched call {call_id} to {payload.phone_number} using agent {payload.agent_id}")
        
    except Exception as e:
        logger.error(f"Failed to dispatch call: {e}")
        if calls_collection is not None:
            calls_collection.update_one(
                {"call_id": call_id},
                {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow()}}
            )
        raise HTTPException(status_code=500, detail="Failed to initiate call")
    
    return CallResponse(call_id=call_id, status="initiated")

@app.patch("/api/call/{call_id}/status")
def update_call_status(call_id: str, status: str = Body(...)):
    """Update the status of a call"""
    if calls_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    result = calls_collection.update_one(
        {"call_id": call_id}, 
        {"$set": {"status": status, "updated_at": datetime.utcnow()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
    
    return {"detail": f"Call {call_id} status updated to {status}"}

@app.get("/api/calls")
def get_calls(skip: int = 0, limit: int = 50):
    """Return all call records, sorted by most recent"""
    if calls_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    calls = list(calls_collection.find().sort("_id", -1).skip(skip).limit(limit))
    for call in calls:
        call["id"] = str(call.get("_id"))
        call.pop("_id", None)
        # Convert datetime fields to isoformat if present
        for k, v in call.items():
            if hasattr(v, 'isoformat'):
                call[k] = v.isoformat()
    
    return calls

@app.get("/api/call/{call_id}")
def get_call(call_id: str):
    """Get call details by ID"""
    if calls_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    call = calls_collection.find_one({"call_id": call_id})
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call["id"] = call.get("call_id")
    call.pop("_id", None)
    # Convert datetime fields to isoformat if present
    for k, v in call.items():
        if hasattr(v, 'isoformat'):
            call[k] = v.isoformat()
    
    return call

# Analytics Routes
@app.get("/api/analytics/dashboard")
def get_dashboard_metrics():
    """Get dashboard analytics"""
    if agents_collection is None or calls_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    # Basic metrics
    total_agents = agents_collection.count_documents({})
    total_calls = calls_collection.count_documents({})
    
    # Recent calls (last 24 hours)
    recent_calls = calls_collection.count_documents({
        "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}
    })
    
    # Success rate calculation
    successful_calls = calls_collection.count_documents({"status": "completed"})
    success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
    
    # RAG status distribution
    rag_status_pipeline = [
        {"$group": {"_id": "$rag_status", "count": {"$sum": 1}}}
    ]
    rag_status_dist = list(agents_collection.aggregate(rag_status_pipeline))
    
    return {
        "total_agents": total_agents,
        "total_calls": total_calls,
        "recent_calls_24h": recent_calls,
        "call_success_rate": round(success_rate, 2),
        "rag_status_distribution": rag_status_dist,
        "timestamp": datetime.utcnow().isoformat()
    }

# Document Management Routes
@app.get("/api/agent/{agent_id}/documents")
def get_agent_documents(agent_id: str):
    """Get all documents for an agent"""
    if documents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    docs = list(documents_collection.find({"agent_id": agent_id}))
    for doc in docs:
        doc["id"] = str(doc.get("_id"))
        doc.pop("_id", None)
        # Convert datetime fields
        for k, v in doc.items():
            if hasattr(v, 'isoformat'):
                doc[k] = v.isoformat()
    
    return docs

@app.delete("/api/agent/{agent_id}/documents/{doc_id}")
def delete_agent_document(agent_id: str, doc_id: str):
    """Delete a specific document from an agent's knowledge base"""
    if documents_collection is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    # Delete from MongoDB
    result = documents_collection.delete_one({"doc_id": doc_id, "agent_id": agent_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete vectors from Pinecone
    if index is not None:
        try:
            namespace = f"agent_{agent_id}"
            # Delete all chunks for this document
            index.delete(filter={"doc_id": doc_id}, namespace=namespace)
        except Exception as e:
            logger.warning(f"Failed to delete Pinecone vectors for document {doc_id}: {e}")
    
    return {"detail": f"Document {doc_id} deleted"}

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    logger.info("Starting AI Agent System API Server")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )