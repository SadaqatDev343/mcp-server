"""
Demo Script for MCP Memory Workshop Server
==========================================

This demo showcases all the features of the Enhanced MCP Memory Workshop Server:
- Memory storage and retrieval
- Session management
- Analytics and insights
- Context extraction
- Relevancy scoring
- Trend analysis

Run this after starting your workshop_server.py
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import random

# Demo Configuration
SERVER_URL = "http://localhost:8081"
DEMO_SESSION_ID = "demo_session_2024"

class MCPMemoryDemo:
    """Comprehensive demo client for MCP Memory Workshop Server."""
    
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_server_status(self) -> Dict[str, Any]:
        """Check if the workshop server is running."""
        try:
            async with self.session.get(f"{self.server_url}/workshop/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Server returned status {response.status}")
        except Exception as e:
            raise Exception(f"Failed to connect to server: {e}")
    
    async def store_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """Store a message in the agent's memory."""
        payload = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        
        async with self.session.post(
            f"{self.server_url}/memory/store",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to store message: {error_text}")
    
    async def recall_memory(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve stored messages for a session."""
        async with self.session.get(
            f"{self.server_url}/memory/recall/{session_id}?limit={limit}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to recall memory: {error_text}")
    
    async def get_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a session."""
        async with self.session.get(
            f"{self.server_url}/memory/analytics/{session_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get analytics: {error_text}")
    
    async def extract_context(self, session_id: str) -> Dict[str, Any]:
        """Extract complete context for a session."""
        async with self.session.get(
            f"{self.server_url}/memory/extract/{session_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to extract context: {error_text}")
    
    async def get_relevancy_scores(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get relevancy scores for messages."""
        async with self.session.get(
            f"{self.server_url}/memory/relevancy/{session_id}?limit={limit}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get relevancy scores: {error_text}")
    
    async def get_context_details(self, session_id: str) -> List[Dict[str, Any]]:
        """Get detailed context information."""
        async with self.session.get(
            f"{self.server_url}/memory/context_details/{session_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get context details: {error_text}")
    
    async def get_trends(self, session_id: str) -> List[Dict[str, Any]]:
        """Get usage trends for a session."""
        async with self.session.get(
            f"{self.server_url}/memory/trends/{session_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get trends: {error_text}")
    
    async def list_sessions(self) -> List[str]:
        """List all available sessions."""
        async with self.session.get(f"{self.server_url}/memory/sessions") as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Failed to list sessions: {error_text}")

# Demo Data
DEMO_CONVERSATION = [
    {"role": "user", "content": "Hello! I'm interested in learning about machine learning. Can you help me get started?"},
    {"role": "assistant", "content": "Absolutely! I'd be happy to help you get started with machine learning. Machine learning is a subset of artificial intelligence that focuses on creating systems that can learn and improve from data without being explicitly programmed for every task. What specific area interests you most - supervised learning, unsupervised learning, or reinforcement learning?"},
    {"role": "user", "content": "I think supervised learning sounds interesting. What are some common algorithms I should know about?"},
    {"role": "assistant", "content": "Great choice! Supervised learning is perfect for beginners. Here are some fundamental algorithms you should know: 1) Linear Regression - for predicting continuous values, 2) Logistic Regression - for binary classification, 3) Decision Trees - intuitive and interpretable, 4) Random Forest - ensemble method that combines multiple trees, 5) Support Vector Machines (SVM) - great for classification tasks, 6) K-Nearest Neighbors (KNN) - simple but effective. Which of these would you like to explore first?"},
    {"role": "user", "content": "Decision trees seem intuitive. How do they work exactly?"},
    {"role": "assistant", "content": "Decision trees work by creating a tree-like model of decisions! Here's how: 1) Start with your dataset at the root node, 2) Find the feature that best splits your data (using metrics like Gini impurity or information gain), 3) Create branches for each possible value of that feature, 4) Repeat this process for each subset of data, 5) Continue until you reach a stopping condition (like maximum depth or minimum samples). The beauty is that you can literally follow the tree like a flowchart to make predictions!"},
    {"role": "user", "content": "That makes sense! What about overfitting? I've heard that's a common problem."},
    {"role": "assistant", "content": "Excellent question! Overfitting is indeed a major concern with decision trees. It happens when your tree becomes too complex and memorizes the training data instead of learning general patterns. Signs of overfitting: perfect accuracy on training data but poor performance on new data. Solutions include: 1) Pruning - removing branches that don't improve performance, 2) Setting maximum depth limits, 3) Requiring minimum samples per leaf, 4) Using ensemble methods like Random Forest. The key is finding the right balance between complexity and generalization."},
    {"role": "user", "content": "How do I know if my model is overfitting? Are there specific metrics to watch?"},
    {"role": "assistant", "content": "Great question! Here are key indicators and metrics to watch for overfitting: 1) **Training vs Validation Performance**: Large gap between training accuracy (high) and validation accuracy (low), 2) **Learning Curves**: Plot training/validation error over time - overfitting shows diverging curves, 3) **Cross-Validation**: Use k-fold CV to get more robust performance estimates, 4) **Metrics to Monitor**: Accuracy, Precision, Recall, F1-score, AUC-ROC. Tools like scikit-learn make this easy with train_test_split and cross_val_score functions."},
    {"role": "user", "content": "This has been really helpful! Can you recommend some good resources for hands-on practice?"},
    {"role": "assistant", "content": "I'm glad you found this helpful! Here are excellent resources for hands-on ML practice: **Free Platforms**: 1) Kaggle - competitions and datasets, 2) Google Colab - free GPU access, 3) Jupyter notebooks, **Courses**: 1) Andrew Ng's Machine Learning Course (Coursera), 2) fast.ai - practical deep learning, **Books**: 1) 'Hands-On Machine Learning' by Aurélien Géron, 2) 'Python Machine Learning' by Sebastian Raschka, **Practice**: Start with simple datasets like Iris or Titanic, then gradually tackle more complex problems. The key is consistent practice and experimenting with real data!"}
]

def print_section(title: str, content: str = ""):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    if content:
        print(content)

def print_json(data: Any, title: str = ""):
    """Print JSON data in a formatted way."""
    if title:
        print(f"\n {title}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

async def run_comprehensive_demo():
    """Run the complete MCP Memory Workshop demo."""
    print(" Starting MCP Memory Workshop Demo")
    print("=" * 60)
    
    async with MCPMemoryDemo() as demo:
        try:
            # 1. Check Server Status
            print_section("Server Status Check")
            status = await demo.check_server_status()
            print(f" Server Status: {status['server_status']}")
            print(f" Workshop: {status['workshop']}")
            print(f" Database: {status['database']}")
            print_json(status['memory_stats'], "Current Memory Stats")
            
            # 2. Store Demo Conversation
            print_section("Storing Demo Conversation")
            print(f" Storing {len(DEMO_CONVERSATION)} messages in session: {DEMO_SESSION_ID}")
            
            for i, msg in enumerate(DEMO_CONVERSATION):
                result = await demo.store_message(
                    DEMO_SESSION_ID, 
                    msg["role"], 
                    msg["content"]
                )
                print(f" Message {i+1}/{len(DEMO_CONVERSATION)} stored: {result['message_id'][:8]}...")
                await asyncio.sleep(0.1)  # Small delay for realistic timing
            
            # 3. Recall Memory
            print_section("Memory Recall Demo")
            recalled = await demo.recall_memory(DEMO_SESSION_ID, limit=5)
            print(f" Retrieved {len(recalled)} most recent messages:")
            for i, msg in enumerate(recalled):
                print(f"\n{i+1}. [{msg['role'].upper()}] {msg['content'][:100]}...")
            
            # 4. Analytics
            print_section("Session Analytics")
            analytics = await demo.get_analytics(DEMO_SESSION_ID)
            print_json(analytics, "Session Analytics")
            
            # 5. Full Context Extraction
            print_section("Full Context Extraction")
            context = await demo.extract_context(DEMO_SESSION_ID)
            print(f" Extracted {context['total_messages']} messages from session")
            print(f" Extraction time: {context['extracted_at']}")
            
            # Show first few messages
            print(f"\n First 3 messages in context:")
            for i, msg in enumerate(context['context'][:3]):
                print(f"{i+1}. [{msg['role'].upper()}] {msg['content'][:80]}...")
            
            # 6. Relevancy Scoring
            print_section("Message Relevancy Analysis")
            relevancy = await demo.get_relevancy_scores(DEMO_SESSION_ID)
            print(f" Analyzed {len(relevancy)} messages for relevancy")
            
            # Sort by relevancy score
            relevancy_sorted = sorted(relevancy, key=lambda x: x['relevancy_score'], reverse=True)
            print(f"\n Top 3 most relevant messages:")
            for i, msg in enumerate(relevancy_sorted[:3]):
                print(f"{i+1}. Score: {msg['relevancy_score']:.3f}")
                print(f"   Keywords: {', '.join(msg['keywords'])}")
                print(f"   ID: {msg['message_id'][:8]}...")
            
            # 7. Context Details
            print_section("Context Details (Debugging)")
            details = await demo.get_context_details(DEMO_SESSION_ID)
            print(f" Retrieved details for {len(details)} messages")
            
            # Show statistics
            total_chars = sum(d['content_length'] for d in details)
            user_msgs = len([d for d in details if d['role'] == 'user'])
            assistant_msgs = len([d for d in details if d['role'] == 'assistant'])
            
            print(f"\n Content Statistics:")
            print(f"   Total characters: {total_chars:,}")
            print(f"   User messages: {user_msgs}")
            print(f"   Assistant messages: {assistant_msgs}")
            print(f"   Average message length: {total_chars // len(details)} characters")
            
            # 8. Usage Trends
            print_section("Usage Trends Analysis")
            trends = await demo.get_trends(DEMO_SESSION_ID)
            print(f" Generated trends for {len(trends)} days")
            
            if trends:
                print_json(trends, "Daily Usage Trends")
            else:
                print("  No trend data available (messages might be from same day)")
            
            # 9. List All Sessions
            print_section("Available Sessions")
            sessions = await demo.list_sessions()
            print(f" Found {len(sessions)} sessions in memory:")
            for i, session in enumerate(sessions):
                print(f"   {i+1}. {session}")
            
            # 10. Demo Complete
            print_section("Demo Complete! ")
            print(" All MCP Memory Workshop features demonstrated successfully!")
            print("\n Try these URLs in your browser:")
            print(f"   • Server Info: {SERVER_URL}/")
            print(f"   • API Docs: {SERVER_URL}/docs")
            print(f"   • Workshop Status: {SERVER_URL}/workshop/status")
            print(f"\n You can now experiment with the API endpoints!")
            
        except Exception as e:
            print(f" Demo failed: {e}")
            print("\n Troubleshooting:")
            print("   1. Make sure workshop_server.py is running")
            print("   2. Check if server is accessible at http://localhost:8081")
            print("   3. Verify no firewall is blocking the connection")

if __name__ == "__main__":
    print(" MCP Memory Workshop Demo")
    print("=" * 40)
    print("Running comprehensive demo...")
    
    asyncio.run(run_comprehensive_demo())