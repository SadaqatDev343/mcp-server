# "Structuring Agent Memory with MCP"
# 
# This workshop demonstrates how to implement persistent memory for AI agents
# using the Model Context Protocol (MCP) pattern, moving beyond simple prompt engineering
# to create truly stateful and context-aware AI interactions.

#region Libraries and Imports       
# FastAPI framework for building the REST API server
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
# Pydantic for data validation and serialization
from pydantic import BaseModel, Field
# Python standard library imports
from typing import List, Dict, Any
from datetime import datetime, timezone
import logging
import sqlite3  # Lightweight database for persistent storage
from pathlib import Path  # Cross-platform path handling
import uuid  # For generating unique message identifiers
from contextlib import asynccontextmanager  # For app lifecycle management
import re  # For text processing in relevancy calculation
#endregion

#region Configuration and Initialization        
# 1. Logging Configuration
# Set up structured logging to track memory operations and server events
# This helps with debugging and monitoring the agent's memory activities
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 2. Database Path and Global Connection
# SQLite database file for persistent storage of conversation memory
# Using a global connection for simplicity in this workshop demonstration
db_path = Path("workshop_memory.db")
db = None  # Will be initialized during app startup

def initialize_workshop_database():
    """
    Initialize SQLite database and create required tables/indexes if not present.
    Ensures efficient storage and retrieval for workshop demonstration.
    """
    global db
    try:
        # Create database connection with WAL mode for better performance
        db = sqlite3.connect(db_path, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
        db.row_factory = sqlite3.Row  # Enable dict-like row access
        
        # Create the main conversations table
        db.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,           -- Unique identifier for each message
                session_id TEXT NOT NULL,      -- Groups messages by conversation session
                timestamp TEXT NOT NULL,       -- When the message was created (ISO format)
                role TEXT NOT NULL,           -- 'user' or 'assistant' to track message source
                content TEXT NOT NULL,        -- The actual message content
                created_at TEXT NOT NULL      -- Database insertion timestamp
            )
        ''')
        
        # Create indexes for efficient querying
        db.execute('''CREATE INDEX IF NOT EXISTS idx_session_timestamp ON conversations(session_id, timestamp)''')
        db.execute('''CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)''')
        
        # Commit the schema changes
        db.commit()
        logger.info(" Workshop database initialized successfully")
        return db
    except Exception as e:
        logger.error(f" Failed to initialize workshop database: {e}")
        raise

# 3. Application Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI app.
    Ensures database is initialized and closed properly.
    """
    global db
    logger.info("Starting MCP Memory Workshop Server...")
    try:
        # Initialize database connection and schema
        db = initialize_workshop_database()
        logger.info(" Workshop server ready for demonstration")
    except Exception as e:
        logger.error(f" Failed to start workshop server: {e}")
        raise
    
    # Yield control to the FastAPI application
    yield
    
    # Cleanup on shutdown
    logger.info(" Shutting down workshop server...")
    if db:
        db.close()
        logger.info(" Database connection closed successfully")

# 4. FastAPI App Initialization
app = FastAPI(
    title="MCP Memory Workshop Server - Enhanced",
    description="Enhanced demonstration server for 'Structuring Agent Memory with MCP: Beyond Prompt Engineering'",
    version="2.0.0-enhanced",
    lifespan=lifespan
)

# 5. Enhanced Data Models
class MemoryMessage(BaseModel):
    """Input model for storing new messages in agent memory."""
    session_id: str = Field(..., min_length=1, max_length=100, description="Conversation session identifier")
    role: str = Field(..., pattern="^(user|assistant)$", description="Message role in conversation")
    content: str = Field(..., min_length=1, max_length=5000, description="Message content to store in memory")

class StoredMessage(BaseModel):
    """Output model for retrieved messages from agent memory."""
    timestamp: str
    role: str
    content: str

class MemoryResponse(BaseModel):
    """Response model for successful memory storage operations."""
    status: str
    message_id: str
    session_id: str
    timestamp: str

class RelevancyResponse(BaseModel):
    """Response model for message relevancy scoring."""
    message_id: str
    relevancy_score: float
    keywords: List[str]

class TrendData(BaseModel):
    """Response model for trend analytics."""
    date: str
    message_count: int
    user_messages: int
    assistant_messages: int

class ContextDetails(BaseModel):
    """Response model for detailed context information."""
    message_id: str
    session_id: str
    timestamp: str
    role: str
    content_length: int
    created_at: str

# 6. Utility Functions
def compute_relevancy(content: str) -> Dict[str, Any]:
    """
    Compute relevancy score and extract keywords from message content.
    
    Args:
        content: Message content to analyze
        
    Returns:
        Dictionary containing relevancy score and keywords
    """
    try:
        # Simple relevancy scoring based on content characteristics
        word_count = len(content.split())
        char_count = len(content)
        
        # Extract keywords (simple implementation)
        words = re.findall(r'\b\w+\b', content.lower())
        keywords = [word for word in words if len(word) > 3][:5]  # Top 5 keywords
        
        # Calculate relevancy score (0-1 scale)
        relevancy_score = min(1.0, (word_count * 0.1 + char_count * 0.001) / 10)
        
        return {
            "score": round(relevancy_score, 3),
            "keywords": keywords,
            "word_count": word_count,
            "char_count": char_count
        }
    except Exception as e:
        logger.error(f" Relevancy computation failed: {e}")
        return {"score": 0.0, "keywords": [], "word_count": 0, "char_count": 0}

# 7. Store Memory Endpoint
@app.post("/memory/store", response_model=MemoryResponse)
async def store_memory(message: MemoryMessage, background_tasks: BackgroundTasks):
    """Stores a message in the agent's memory for a given session."""
    try:
        # Validate and clean input data
        session_id = message.session_id.strip()
        content = message.content.strip()
        
        if not session_id or not content:
            raise HTTPException(status_code=400, detail="Session ID and content are required")
        
        # Generate unique identifiers and timestamps
        message_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Insert into database
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO conversations (id, session_id, timestamp, role, content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, timestamp, message.role, content, timestamp))
        
        db.commit()
        
        # Background logging
        background_tasks.add_task(
            logger.info,
            f" Memory stored: session={session_id}, role={message.role}, length={len(content)}"
        )
        
        return MemoryResponse(
            status="stored",
            message_id=message_id,
            session_id=session_id,
            timestamp=timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Memory storage error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")

# 8. Recall Memory Endpoint
@app.get("/memory/recall/{session_id}", response_model=List[StoredMessage])
async def recall_memory(
    session_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Number of messages to recall")
):
    """Retrieves the most recent messages for a session."""
    try:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        cursor = db.cursor()
        cursor.execute('''
            SELECT timestamp, role, content 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        messages = cursor.fetchall()
        memory = [StoredMessage(timestamp=row['timestamp'], role=row['role'], content=row['content']) 
                 for row in reversed(messages)]
        
        logger.info(f" Memory recalled: session={session_id}, messages={len(memory)}")
        return memory
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Memory recall error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to recall memory: {str(e)}")

# 9. List Sessions Endpoint
@app.get("/memory/sessions", response_model=List[str])
async def list_sessions(limit: int = Query(default=20, ge=1, le=100, description="Max sessions to list")):
    """Lists all unique session IDs."""
    try:
        cursor = db.cursor()
        cursor.execute('''
            SELECT DISTINCT session_id 
            FROM conversations 
            ORDER BY session_id 
            LIMIT ?
        ''', (limit,))
        
        sessions = [row['session_id'] for row in cursor.fetchall()]
        logger.info(f" Sessions listed: {len(sessions)}")
        return sessions
        
    except Exception as e:
        logger.error(f" Session listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

# 10. Memory Analytics Endpoint
@app.get("/memory/analytics/{session_id}")
async def memory_analytics(session_id: str):
    """Provides comprehensive analytics for a session."""
    try:
        cursor = db.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM conversations 
            WHERE session_id = ?
        ''', (session_id,))
        
        stats = cursor.fetchone()
        
        if stats['total_messages'] == 0:
            return {"session_id": session_id, "status": "no_memory", "total_messages": 0}
        
        # Calculate conversation duration
        first_time = datetime.fromisoformat(stats['first_message'])
        last_time = datetime.fromisoformat(stats['last_message'])
        duration = (last_time - first_time).total_seconds()
        
        analytics = {
            "session_id": session_id,
            "status": "analyzed",
            "total_messages": stats['total_messages'],
            "user_messages": stats['user_messages'],
            "assistant_messages": stats['assistant_messages'],
            "conversation_duration_seconds": duration,
            "first_message": stats['first_message'],
            "last_message": stats['last_message'],
            "avg_response_ratio": round(stats['assistant_messages'] / max(stats['user_messages'], 1), 2)
        }
        
        logger.info(f" Analytics generated for session: {session_id}")
        return analytics
        
    except Exception as e:
        logger.error(f" Memory analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze memory: {str(e)}")

# 11. NEW: Extract Full Context Endpoint
@app.get("/memory/extract/{session_id}")
async def extract_context(session_id: str):
    """
    Extract complete session context with all message details.
    Provides full transparency into stored conversation data.
    """
    try:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        cursor = db.cursor()
        cursor.execute('''
            SELECT id, session_id, timestamp, role, content, created_at 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        rows = cursor.fetchall()
        context = [dict(row) for row in rows]
        
        logger.info(f"📤 Context extracted: session={session_id}, messages={len(context)}")
        return {
            "session_id": session_id,
            "total_messages": len(context),
            "context": context,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f" Context extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract context: {str(e)}")

# 12. NEW: Message Relevancy Logic Endpoint
@app.get("/memory/relevancy/{session_id}", response_model=List[RelevancyResponse])
async def message_relevancy(
    session_id: str, 
    limit: int = Query(default=20, ge=1, le=100, description="Number of messages to analyze")
):
    """
    Expose message relevancy scoring logic for transparency.
    Shows how messages are ranked for contextual importance.
    """
    try:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        cursor = db.cursor()
        cursor.execute('''
            SELECT id, content 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        messages = cursor.fetchall()
        relevancy_results = []
        
        for msg in messages:
            relevancy_data = compute_relevancy(msg['content'])
            relevancy_results.append(RelevancyResponse(
                message_id=msg['id'],
                relevancy_score=relevancy_data['score'],
                keywords=relevancy_data['keywords']
            ))
        
        logger.info(f" Relevancy computed: session={session_id}, messages={len(relevancy_results)}")
        return relevancy_results
        
    except Exception as e:
        logger.error(f" Relevancy computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compute relevancy: {str(e)}")

# 13. NEW: Context Details for Debugging
@app.get("/memory/context_details/{session_id}", response_model=List[ContextDetails])
async def context_details(session_id: str):
    """
    Retrieve detailed context information for debugging and transparency.
    Shows metadata about each message without full content.
    """
    try:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        cursor = db.cursor()
        cursor.execute('''
            SELECT id, session_id, timestamp, role, LENGTH(content) as content_length, created_at
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        
        details = [
            ContextDetails(
                message_id=row['id'],
                session_id=row['session_id'],
                timestamp=row['timestamp'],
                role=row['role'],
                content_length=row['content_length'],
                created_at=row['created_at']
            )
            for row in cursor.fetchall()
        ]
        
        logger.info(f" Context details retrieved: session={session_id}, messages={len(details)}")
        return details
        
    except Exception as e:
        logger.error(f" Context details failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve context details: {str(e)}")

# 14. NEW: Context Usage Trends
@app.get("/memory/trends/{session_id}", response_model=List[TrendData])
async def context_trends(session_id: str):
    """
    Display trends in context usage over time for a session.
    Shows conversation activity patterns and usage analytics.
    """
    try:
        session_id = session_id.strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        cursor = db.cursor()
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count,
                COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages
            FROM conversations 
            WHERE session_id = ? 
            GROUP BY DATE(timestamp) 
            ORDER BY date ASC
        ''', (session_id,))
        
        trends = [
            TrendData(
                date=row['date'],
                message_count=row['message_count'],
                user_messages=row['user_messages'],
                assistant_messages=row['assistant_messages']
            )
            for row in cursor.fetchall()
        ]
        
        logger.info(f" Trends generated: session={session_id}, days={len(trends)}")
        return trends
        
    except Exception as e:
        logger.error(f" Trends generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trends: {str(e)}")

# 15. Enhanced Workshop Status Endpoint
@app.get("/workshop/status")
async def workshop_status():
    """Returns enhanced server status and memory statistics."""
    try:
        cursor = db.cursor()
        
        # Gather comprehensive statistics
        cursor.execute('SELECT COUNT(*) as total FROM conversations')
        total_messages = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(DISTINCT session_id) as total FROM conversations')
        total_sessions = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as total FROM conversations WHERE role = "user"')
        user_messages = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as total FROM conversations WHERE role = "assistant"')
        assistant_messages = cursor.fetchone()['total']
        
        return {
            "workshop": "Structuring Agent Memory with MCP: Beyond Prompt Engineering",
            "server_status": "active",
            "database": str(db_path),
            "memory_stats": {
                "total_messages": total_messages,
                "total_sessions": total_sessions,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages
            },
            "capabilities": [
                "Persistent conversation memory",
                "Session-based organization", 
                "Memory recall and analysis",
                "Full context extraction",
                "Message relevancy scoring",
                "Context usage trends",
                "Debugging transparency"
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f" Status check error: {e}")
        return {
            "workshop": "Structuring Agent Memory with MCP: Beyond Prompt Engineering",
            "server_status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# 16. Enhanced Root Endpoint
@app.get("/")
async def workshop_info():
    """Returns comprehensive workshop metadata and API documentation."""
    return {
        "title": "Enhanced MCP Memory Workshop Server",
        "workshop": "Structuring Agent Memory with MCP: Beyond Prompt Engineering",
        "description": "Enhanced demonstration server with full context extraction and transparency features",
        "version": "2.0.0-enhanced",
        "endpoints": {
            "store_memory": "POST /memory/store - Store conversation messages",
            "recall_memory": "GET /memory/recall/{session_id} - Retrieve conversation history",
            "list_sessions": "GET /memory/sessions - List all conversation sessions",
            "memory_analytics": "GET /memory/analytics/{session_id} - Analyze conversation patterns",
            "extract_context": "GET /memory/extract/{session_id} - Extract full session context",
            "message_relevancy": "GET /memory/relevancy/{session_id} - Message relevancy scoring",
            "context_details": "GET /memory/context_details/{session_id} - Debugging context details",
            "context_trends": "GET /memory/trends/{session_id} - Context usage trends",
            "workshop_status": "GET /workshop/status - Server and memory overview"
        },
        "key_concepts": [
            "Structured memory storage",
            "Persistent context across sessions",
            "Session-based organization",
            "Memory analysis and insights",
            "Context extraction transparency",
            "Message relevancy scoring",
            "Usage trend analytics"
        ],
        "documentation": "http://localhost:8081/docs",
        "database": str(db_path)
    }

# 17. Server Startup
if __name__ == "__main__":
    import uvicorn
    import socket

    def find_workshop_port():
        """Find an available port for the workshop server."""
        preferred_ports = [8081, 8080, 8888, 9000]
        
        for port in preferred_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 8081

    port = find_workshop_port()
    
    print("Enhanced MCP Memory Workshop Server")
    print("=" * 60)
    print(f" Topic: Structuring Agent Memory with MCP: Beyond Prompt Engineering")
    print(f" Server: http://localhost:{port}")
    print(f" API Docs: http://localhost:{port}/docs")
    print(f" Database: {db_path}")
    print(f" New Features: Context extraction, relevancy scoring, trends")
    print(" Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False
    )