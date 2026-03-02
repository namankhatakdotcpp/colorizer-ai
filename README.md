# AI Colorizer Platform

A production-grade, full-stack AI application that converts grayscale images to color using a Convolutional Neural Network (CNN) in the LAB color space.

## Architecture

- **Backend**: FastAPI + PyTorch CNN
- **Frontend**: Next.js (App Router) + Tailwind CSS + Lucide Icons
- **Infrastructure**: Docker, Docker Compose, GitHub Actions CI/CD

## Running Locally

### Prerequisites
- Docker & Docker Compose
- Node.js 20 (optional, for frontend local dev)
- Python 3.11 (optional, for backend local dev)

### Using Docker (Recommended)

1. Clone the repository
2. Run `docker-compose up --build`
3. Access the frontend at `http://localhost:3000`
4. Access the API at `http://localhost:8000/docs`

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Model Architecture

The colorization model (`backend/app/model/colorizer.py`) processes images in the **LAB color space**:
- Input: L channel (Lightness/Grayscale)
- Output: "A" and "B" channels (Color)
- Arch: Encoder-Decoder CNN structure with Batch Normalization.

*(Note: In this boilerplate, the model uses random weights until trained on a high-fidelity dataset like ImageNet)*.

## CI/CD 
GitHub Actions are configured in `.github/workflows/ci.yml` to automatically lint and test the FastAPI backend and build the Next.js app on pull requests and pushes to `main`.
