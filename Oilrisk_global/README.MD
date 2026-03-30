# OilRisk — Global Oil Supply Risk Prediction

## Worldwide Conflict-Zone Oil Disruption Intelligence Platform

A full-stack Flask web application using **Bagging & Boosting ML models** to predict
oil supply disruption risk from global conflict events — covering Israel–Iran,
Red Sea / Houthis, Russia–Ukraine, Libya, Nigeria, Venezuela, South China Sea, Sudan, and more.

---

## What's New (v3.0 — Global Edition)

| Feature           | v2.0 (Israel–Iran only) | v3.0 (Global)                                            |
| ----------------- | ----------------------- | -------------------------------------------------------- |
| Dataset coverage  | 1 region                | **7 regions, 46+ locations**                             |
| Actor coverage    | 5 attackers             | **19 attackers worldwide**                               |
| Event types       | 6                       | **9 (incl. Pipeline_Sabotage, Mine_Attack)**             |
| Target types      | 7                       | **12 (incl. Oil_Tanker, LNG_Terminal, Strait_Blockade)** |
| UI theme          | Dark geopolitical       | **Modern light glassmorphism**                           |
| Live world map    | ✗                       | **✅ Animated canvas world map**                         |
| Risk ticker       | ✗                       | **✅ Live scrolling news ticker**                        |
| Region filter     | ✗                       | **✅ Dataset explorer region filter**                    |
| Region risk cards | ✗                       | **✅ 8 region risk summary cards**                       |
| Charts            | 2                       | **4 on dashboard**                                       |

---

## Project Structure

```
flask_app/
├── app.py                      ← Flask application — global edition
├── main.py                     ← CLI pipeline (unchanged interface)
├── setup.py                    ← Package setup
├── requirements.txt            ← Dependencies
├── src/
│   ├── __init__.py
│   ├── data_generator.py       ← Global 1500-row dataset (46+ locations, 7 regions)
│   ├── preprocess.py           ← Cleaning, encoding, scaling
│   ├── train_models.py         ← 6 ML models (Decision Tree → XGBoost)
│   ├── evaluate.py             ← Evaluation, charts, Excel export
│   └── predict.py              ← Single & batch real-time prediction
├── templates/
│   ├── base.html               ← Glassmorphism shell + animated background
│   ├── dashboard.html          ← Live map + ticker + model results
│   ├── predict.html            ← Single-event prediction (4 quick-fill presets)
│   ├── batch_predict.html      ← JSON batch prediction
│   ├── visualizations.html     ← Chart gallery
│   ├── dataset.html            ← Paginated explorer with region filter
│   └── sustainability.html     ← Oil stock sustainability calculator
├── static/
│   ├── css/style.css           ← Light glassmorphism theme (Syne + DM Sans fonts)
│   └── js/live-tracker.js      ← Animated world-map canvas renderer
├── data/                       ← Auto-created: Global_Oil_Risk_dataset.csv
└── outputs/                    ← Auto-created: PNGs + Excel workbook
```

---

## Regions Covered

| Region           | Key Locations                                                | Risk Focus            |
| ---------------- | ------------------------------------------------------------ | --------------------- |
| **Middle East**  | Tehran, Natanz, Kharg Island, Strait of Hormuz, Saudi Aramco | Iran nuclear + oil    |
| **Red Sea**      | Bab-el-Mandeb, Suez Canal, Aden Gulf                         | Houthi tanker attacks |
| **Europe**       | Kyiv, Zaporizhzhia, Novorossiysk, Baltic Pipeline            | Russia–Ukraine        |
| **Africa**       | Lagos, Niger Delta, Tripoli, Sirte Basin                     | Nigeria, Libya        |
| **Americas**     | Caracas, Maracaibo, Orinoco Belt                             | Venezuela             |
| **Asia-Pacific** | South China Sea, Strait of Malacca, Spratly Islands          | China tension         |
| **Caucasus**     | Baku, BTC Pipeline                                           | Azerbaijan            |

---

## Setup & Run

```bash
# 1. Clone / copy project
cd flask_app

# 2. Install dependencies
pip install -r requirements.txt

# Optional: install in editable mode
pip install -e .

# 3. Start Flask
python app.py

# 4. Open browser
http://localhost:5000
```

---

## Pages & Features

| URL               | Description                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| `/`               | Dashboard — live world map, risk ticker, region cards, pipeline control, model results |
| `/predict`        | Single-event prediction — 4 quick-fill presets (Hormuz, Red Sea, Ukraine, Low)         |
| `/batch-predict`  | JSON batch prediction for 5 worldwide sample events                                    |
| `/visualizations` | EDA, correlation, feature importance, model comparison, confusion matrices             |
| `/dataset`        | Paginated 1,500-row explorer with search + risk + **region** filters                   |
| `/sustainability` | Oil stock sustainability calculator — 5 disruption scenarios                           |

## JSON API

| Endpoint                           | Returns                               |
| ---------------------------------- | ------------------------------------- |
| `GET /api/stats`                   | Dataset summary + region distribution |
| `GET /api/model-results`           | Model accuracy comparison             |
| `GET /api/sustainability`          | Disruption scenario table             |
| `GET /api/risk-distribution`       | Risk label counts                     |
| `GET /api/region-distribution`     | Events by region                      |
| `GET /api/event-type-distribution` | Event type counts                     |
| `GET /api/live-events`             | Simulated live risk events for map    |
| `GET /health`                      | App + pipeline status                 |

---

## ML Models

| #   | Model             | Family         |
| --- | ----------------- | -------------- |
| 1   | Decision Tree     | Bagging (base) |
| 2   | Random Forest     | Bagging        |
| 3   | BaggingClassifier | Bagging        |
| 4   | AdaBoost          | Boosting       |
| 5   | Gradient Boosting | Boosting       |
| 6   | XGBoost           | Boosting       |

All models are hyperparameter-tuned with **GridSearchCV (3-fold CV)**.

---

## Design System

- **Theme**: Light glassmorphism — `rgba(255,255,255,0.72)` cards, `backdrop-filter: blur(20px)`
- **Fonts**: Syne (display/headings) + DM Sans (body)
- **Palette**: Primary `#1a56e8` · Accent `#f97316` · Low `#10b981` · Medium `#f59e0b` · High `#ef4444` · Critical `#8b5cf6`
- **Background**: Animated gradient mesh + moving grid + floating orbs
- **Live elements**: Canvas world map with ripple hotspots · Scrolling risk ticker

---

## Notes

- `src/` modules have a stable interface — `app.py` only wraps them
- Pipeline runs in a background thread; UI streams live log output via SSE
- XGBoost is optional; falls back to GradientBoostingClassifier if not installed
- Dataset filename changed from `Israel_Iran_oil_dataset.csv` → `Global_Oil_Risk_dataset.csv`

---

## Map — Leaflet.js + Esri Satellite (v6.0)

**100% Free — No API key, No payment, No signup required.**

Uses [Leaflet.js](https://leafletjs.com/) (open-source) with free tile providers:

| Map Type      | Provider           | Cost |
| ------------- | ------------------ | ---- |
| 🛰️ Satellite  | Esri World Imagery | Free |
| 🗺️ Street Map | OpenStreetMap      | Free |
| 🏔️ Terrain    | OpenTopoMap        | Free |
| 🌑 Dark Mode  | CartoDB Dark       | Free |

### Setup (no configuration needed)

```bash
python app.py
# Map works immediately — open http://localhost:5000
```

### Map Features

| Feature               | Description                                                                             |
| --------------------- | --------------------------------------------------------------------------------------- |
| **4 base map types**  | Satellite · Street · Terrain · Dark — switch top-right                                  |
| **27 custom markers** | Pulsing SVG markers colour-coded by risk level                                          |
| **Click popup**       | Rich popup: flag · country · risk badge · description                                   |
| **Risk circles**      | Translucent exposure-radius overlays per hotspot                                        |
| **Threat arcs**       | Dashed curved polylines connecting risk zones                                           |
| **Risk filter pills** | Filter markers by CRITICAL / HIGH / MEDIUM / LOW                                        |
| **Layer toggles**     | Show/hide circles and arcs independently                                                |
| **Region jump bar**   | One-click fly-to: Gulf · Israel–Iran · Red Sea · Ukraine · India · S.China Sea · Africa |
| **Side panel**        | Clicked marker details update the right panel                                           |
| **Scale bar**         | Distance scale shown bottom-left                                                        |
| **Mobile ready**      | Pinch-to-zoom and drag built into Leaflet                                               |
