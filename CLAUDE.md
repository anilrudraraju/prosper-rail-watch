# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Prosper Rail Watch — a 24/7 AI-powered railroad crossing monitor for two TX-121 crossings in Prosper, TX. A headless Chrome browser screenshots live camera feeds every N seconds; a custom YOLOv11 model detects trains; results are stored in SQLite and served via a Flask REST API to a single-page HTML dashboard.

## Running

```bash
python3 screen_capture_detector.py
```

No build step. No test suite. The app runs as a `systemd` service on a DigitalOcean Ubuntu server.

### Deploying to server

```bash
# From local Mac
scp ~/Downloads/screen_capture_detector.py root@SERVER_IP:/var/www/prosper-rail-watch/

# On server — backup, restart, verify
cp screen_capture_detector.py screen_capture_detector.py.bak.vX.X
systemctl restart prosper-rail-watch
sleep 35 && cat /var/log/prosper-autostart.log
# Expected: {"status":"running","success":true}
```

### Dependencies

```bash
pip install -r requirements.txt
# selenium, webdriver-manager, Pillow, numpy, opencv-python, flask, flask-cors, pytz, ultralytics
```

## Architecture

### Data flow

```
Two camera threads (staggered 30s apart)
  → Selenium + headless Chrome (fresh open/close every cycle — prevents ipcamlive freeze)
  → YOLOv11 model per camera
  → SQLite (detections + train_events tables)
  → Flask REST API (port 5000)
  → simple-rail-watch.html (polls every 30s)
```

### Key files

| File | Purpose |
|---|---|
| `screen_capture_detector.py` | Everything: detection engine, scheduler, DB, Flask API (~1016 lines) |
| `simple-rail-watch.html` | Frontend dashboard — vanilla JS + Chart.js, no build |
| `requirements.txt` | Python dependencies |

Not in repo (live data / too large): `*.pt` model weights, `detections.db`, `screenshots/`, `training_capture/`.

### Scheduling logic

- **Active hours** (7:00 AM – 7:00 PM CDT): 20s capture interval
- **Blackout** (7:00 PM – 7:00 AM CDT): both cameras sleep
- **Burst mode**: train detected → 60s interval for 10 minutes to track movement
- **Training mode**: pauses normal monitoring on one camera to collect labeled frames

### Detection thresholds

- `≥ 70%` confidence → `train_detected = True`
- `≥ 20%` confidence → possible train / triggers burst mode
- `< 20%` → ends an active train event

### Database schema (SQLite)

**`detections`** — one row per capture: `camera_id`, `timestamp`, `train_detected`, `confidence`, `detection_details` (JSON), `screenshot_path`, `is_school_day`

**`train_events`** — aggregated event: `start_time`, `end_time`, `duration_seconds`, `max_confidence`, plus trend fields (`day_of_week`, `hour_of_day`, `month`, `week_number`)

Use `get_db_connection()` context manager for all DB access.

### Flask API endpoints

| Endpoint | Notes |
|---|---|
| `GET /api/status` | System status + camera list |
| `GET /api/detections/recent` | Recent checks; `?train_only=true` for history tab |
| `GET /api/stats?days=N` | Aggregated statistics |
| `GET /api/train-events` | Train events with duration |
| `GET /api/trends` | Day-of-week / hourly / weekly aggregations |
| `GET /api/detections/export` | CSV download |
| `POST /api/training_capture` | Start training image collection |
| `GET /api/training_capture` | Training status |
| `POST /api/training_capture/stop` | Stop training |
| `POST /api/start` / `POST /api/stop` | Start/stop monitoring |

### Important implementation details

**Why Chrome opens/closes every cycle**: keeping the browser open causes ipcamlive's player to freeze after ~30 min, serving the same static frame. Fresh open/close every cycle is the fix.

**Chrome lock** (`chrome_lock`): a threading lock that serializes browser opens to prevent crashing the 1-vCPU server.

**Class name normalization**: YOLO outputs `'Train'` but all comparisons use `.lower()` → `'train'`.

**Screenshot layout**: `screenshots/{camera_name}/{trains_detected|no_trains|possible_trains}/` — each image has a paired `.json` metadata file.

**Confidence scale**: YOLO outputs 0–100 (already multiplied by 100) for display and DB storage.

**Logging style**: emoji-prefixed (`🚂` train, `📸` capture, `💾` save, `❌` error) — match this style when adding log lines.

**Timezone**: all times in `America/Chicago` (CDT) via `pytz`.
