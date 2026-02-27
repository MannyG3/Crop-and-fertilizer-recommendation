# Crop & Fertilizer Recommendation System 🌾

A full-stack smart agriculture platform that combines machine learning and modern web technologies to help farmers make data-driven decisions for crop selection and soil nutrient planning.

This project uses trained ML models for crop and fertilizer intelligence, exposes prediction services through FastAPI, and supports MERN-based application workflows for profile management, historical activity, and dashboard visualizations.

---

## Features

- ML-driven crop recommendation using soil and climate parameters (N, P, K, temperature, humidity, pH, rainfall)
- Confidence-scored AI prediction responses for better decision visibility
- Fertilizer recommendation model support for nutrient planning
- FastAPI inference endpoint (`/predict`) with request validation via Pydantic
- Farmer profile and soil history data modeling with MongoDB + Mongoose
- Recent activity retrieval for chart-driven analytics (last 5 recommendations)
- Frontend-ready UI components (Tailwind + Chart.js radar visualization for N-P-K comparison)

---

## Tech Stack

### Frontend (MERN Layer)

- React.js
- Tailwind CSS
- Chart.js (`react-chartjs-2`)

### Backend (MERN Layer)

- Node.js
- Express.js
- MongoDB
- Mongoose

### AI/ML Inference Layer

- Python
- FastAPI
- Scikit-learn
- Joblib
- NumPy
- Pydantic

---

## Project Structure

```text
Crop-and-fertilizer-recommendation/
├── app/
│   └── main.py                      # FastAPI crop prediction service
├── backend/
│   ├── models/
│   │   └── farmerSchemas.js         # FarmerProfile + SoilHistory schemas
│   ├── server.js                    # Express server and recent-activity route
│   ├── package.json
│   └── .env.example
├── src/
│   └── components/
│       ├── SoilInputForm.jsx
│       └── NPKRadarChart.jsx
├── model/
│   ├── crop_model.pkl
│   ├── crop_scaler.pkl
│   ├── fertilizer_model.pkl
│   └── fertilizer_model_optimized.pkl
├── data/
│   ├── Crop_recommendation.csv
│   └── Fertilizer Prediction.csv
└── requirements.txt
```

---

## Setup Instructions

### 1) Clone Repository

```bash
git clone https://github.com/MannyG3/Crop-and-fertilizer-recommendation.git
cd Crop-and-fertilizer-recommendation
```

### 2) Setup Python/FastAPI Service

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- API base URL: `http://127.0.0.1:8000`
- Health check: `GET /health`
- Prediction endpoint: `POST /predict`

Example request body:

```json
{
  "nitrogen": 90,
  "phosphorus": 42,
  "potassium": 43,
  "temperature": 25.0,
  "humidity": 80.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

### 3) Setup Node/Express Backend

```bash
cd backend
cp .env.example .env
npm install
npm start
```

- Backend server default: `http://localhost:4000`
- Recent activity endpoint: `GET /farmers/:farmerId/recent-activity`

### 4) Frontend Integration Notes

If you are building/continuing the React app, install chart dependencies:

```bash
npm install chart.js react-chartjs-2
```

Use the provided components from `src/components` for soil data input and radar chart visualization.

---

## API Snapshot

### FastAPI

- `GET /health` → service status
- `POST /predict` → crop recommendation + confidence

Error behavior:
- Returns `503` with `Model Not Found` when model file is missing
- Returns `500` on model load or prediction failure

### Express

- `GET /farmers/:farmerId/recent-activity` → last 5 AI recommendations for dashboard charting

---

## Future Scope

- Integrate real-time weather APIs (temperature, humidity, rainfall forecasts) to improve prediction context
- Connect IoT soil sensor streams for live N-P-K, moisture, and pH ingestion
- Add seasonality-aware recommendation logic and region-specific crop constraints
- Enable multilingual advisory reports and offline-first farmer workflows
- Introduce model monitoring and automated retraining pipelines

---

## Contributing

Contributions are welcome. Please open an issue for feature requests or bug reports, and submit PRs with clear descriptions and test steps.

---

## License

This project is licensed under the MIT License.
