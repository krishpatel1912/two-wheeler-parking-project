#!/usr/bin/env bash
# ==============================================================================
# ParkWatch AI — Unified Launcher (Backend + Frontend)
# ==============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "🚀 Starting Two-Wheeler Parking Detection System"
echo "=========================================================="

# Ensure virtualenv python exists
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment and installing dependencies..."
    python3 -m venv .venv
    .venv/bin/pip install -r backend/requirements.txt
    .venv/bin/pip install ultralytics opencv-python-headless
fi

# Function to kill child processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $(jobs -p) 2>/dev/null || true
    echo "✓ Stopped cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend API
echo "Starting FastAPI Backend (Port 8000)..."
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait briefly for backend to initialize
sleep 2

# 2. Start Frontend UI
echo "Starting React Frontend (Port 3000)..."
npm run dev --prefix frontend &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo "  ✓ All services running!"
echo "  👉 Open Frontend Dashboard:  http://localhost:3000"
echo "  👉 Backend API Documentation: http://127.0.0.1:8000/docs"
echo "=========================================================="
echo "Press Ctrl+C to stop both servers at any time."
echo ""

# Keep running
wait
