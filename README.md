# Pulse Finance AI Backend

Python backend using FastAPI + FreeFlow LLM for unlimited AI API calls.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate it:
```bash
# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your API keys (JSON arrays for multiple keys)
# GROQ_API_KEY=["gsk_key1", "gsk_key2"]
# GEMINI_API_KEY=["AIzaSy..."]
```

5. Run the server:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/parse-sms` | POST | Parse SMS using AI |
| `/api/explain-spending` | POST | Generate spending insights |
| `/api/infer-category` | POST | Infer transaction category |

## API Docs

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

### Local Development
```bash
python main.py
```

### Production (Docker)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Run / Railway / Render
Just connect the repo and set environment variables.
