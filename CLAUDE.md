# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Commands

### Running the Application
```bash
./run.sh
```
Or manually:
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Installing Dependencies
```bash
uv sync
```

### Environment Setup
Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

### Backend Structure (`backend/`)

- **[app.py](backend/app.py)** - FastAPI application entry point. Serves API endpoints (`/api/query`, `/api/courses`) and static frontend files.
- **[rag_system.py](backend/rag_system.py)** - Main orchestrator that coordinates all components. Initialize `RAGSystem` with config to get a fully functional system.
- **[vector_store.py](backend/vector_store.py)** - ChromaDB wrapper managing two collections:
  - `course_catalog` - Course metadata (titles, instructors, lesson lists)
  - `course_content` - Text chunks for semantic search
- **[document_processor.py](backend/document_processor.py)** - Parses course documents (expects specific format with "Course Title:", "Lesson N:" markers) and creates text chunks.
- **[ai_generator.py](backend/ai_generator.py)** - Claude API integration with tool support. The AI can call `search_course_content` tool to retrieve relevant content.
- **[search_tools.py](backend/search_tools.py)** - Tool definitions for Anthropic tool use. `CourseSearchTool` provides semantic search with optional course/lesson filtering.
- **[session_manager.py](backend/session_manager.py)** - Manages conversation history for multi-turn queries.
- **[config.py](backend/config.py)** - Centralized configuration dataclass with all settings.
- **[models.py](backend/models.py)** - Pydantic models: `Course`, `Lesson`, `CourseChunk`.

### Frontend Structure (`frontend/`)

Simple static HTML/CSS/JS served by FastAPI. No build step required.

### Course Documents (`docs/`)

Plain text files with expected format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [Lesson Title]
Lesson Link: [url]
[content...]

Lesson 1: [Lesson Title]
...
```

## Data Flow

1. **Document Ingestion**: `DocumentProcessor` parses course files → creates `Course` + `CourseChunk` objects → stored in `VectorStore`
2. **Query Flow**: User query → `RAGSystem.query()` → `AIGenerator` with `CourseSearchTool` → Claude decides whether to search → returns answer with sources
3. **Session Management**: `SessionManager` tracks conversation history per session ID

## Key Configuration (config.py)

- `CHUNK_SIZE`: 800 characters per text chunk
- `CHUNK_OVERLAP`: 100 characters overlap between chunks
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation exchanges remembered
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2