# Structuring Agent Memory with MCP: Beyond Prompt Engineering

This project demonstrates how to implement persistent memory for AI agents using the Model Context Protocol (MCP) pattern, moving beyond simple prompt engineering to create truly stateful and context-aware AI interactions. It features a FastAPI server for managing conversation memory and a Python client for demonstration.

## Features

The server provides a robust set of API endpoints for managing and analyzing agent memory:

*   **`/memory/store` (POST)**: Stores a new message in the agent's memory for a given session.
*   **`/memory/recall/{session_id}` (GET)**: Retrieves the most recent messages for a specific session.
*   **`/memory/sessions` (GET)**: Lists all unique conversation session IDs.
*   **`/memory/analytics/{session_id}` (GET)**: Provides comprehensive analytics for a session, including message counts, duration, and response ratios.
*   **`/memory/extract/{session_id}` (GET)**: Extracts the complete session context with all message details for full transparency.
*   **`/memory/relevancy/{session_id}` (GET)**: Exposes message relevancy scoring logic, showing how messages are ranked for contextual importance.
*   **`/memory/context_details/{session_id}` (GET)**: Retrieves detailed context information for debugging, showing metadata about each message without full content.
*   **`/memory/trends/{session_id}` (GET)**: Displays trends in context usage over time for a session, showing conversation activity patterns.
*   **`/workshop/status` (GET)**: Returns enhanced server status and memory statistics.
*   **`/` (GET)**: Provides comprehensive workshop metadata and API documentation.

## Project Structure

```
.
├── workshop_server.py
├── demo_script.py
├── requirements.txt
```

*   `workshop_server.py`: The main FastAPI application that handles memory storage, retrieval, and analysis.
*   `demo_script.py`: A Python client script to interact with the `workshop_server.py` and demonstrate its features.
*   `requirements.txt`: Lists all Python dependencies required to run the project.


## Technologies Used

*   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.8+.
*   **Pydantic**: Data validation and settings management using Python type hints.
*   **Uvicorn**: An ASGI server for running FastAPI applications.
*   **SQLite**: A lightweight, file-based relational database used for persistent memory storage.
*   **aiohttp**: An asynchronous HTTP client/server framework for Python, used in the demo script.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.8 or higher

### Installation

1.  **Clone the repository** (if you haven't already, or create a new one):
    ```bash
    git clone https://github.com/your-username/mcp-memory-workshop.git
    cd mcp-memory-workshop
    ```
    (Replace `your-username/mcp-memory-workshop` with your actual GitHub repository path after pushing.)

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server

1.  **Start the FastAPI server**:
    Make sure your virtual environment is activated.
    ```bash
    python workshop_server.py
    ```
    The server will typically start on `http://localhost:8081`. You can access the API documentation at `http://localhost:8081/docs`. Keep this terminal open.

### Running the Demo

1.  **Ensure the server is running** in a separate terminal.
2.  **Open a new terminal** and activate the virtual environment.
3.  **Run the demo script**:
    ```bash
    python workshopdemo.py
    ```
    This script will interact with the running server, store a sample conversation, and demonstrate all the memory recall and analysis features.

