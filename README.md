# 🚂 Prosper Rail Watch

**Real-time AI train detection for TX-121 crossings in Prosper, TX**  
**Live at [prosperrailwatch.com](https://prosperrailwatch.com)**

Built by **Aarush Rudraraju** — age 15, freshman at Prosper High School (Class of 2029)

---

## The Problem

Two railroad crossings on TX-121 in Prosper regularly block traffic with zero advance warning — affecting hundreds of families, including students at Reynolds Middle School and Prosper High School during morning drop-off and afternoon pickup. I got an official school tardy because of a blocked crossing. So I built a fix.

---

## What It Does

Prosper Rail Watch monitors two crossings 24/7 and tells you **CLEAR** or **TRAIN** before you leave home:

| Crossing | Camera Alias | Status |
|---|---|---|
| TX-121 & Prosper Trail | `prospertrl` | ✅ Active |
| TX-121 & First Street | `firstst` | 🔄 Model retraining (camera repositioned Apr 2026) |

**Detection pipeline:**
1. Selenium + headless Chrome opens the city's public camera web player
2. Screenshot taken after ~10s video buffer load
3. Custom YOLOv11 model runs inference on the screenshot
4. Result (TRAIN/CLEAR + confidence %) saved to SQLite database
5. Website polls the Flask API every 30 seconds and updates live

**Full detection cycle: ~53 seconds** (bottleneck is Chrome page load — see Future Work)

---

## Accuracy

| Model | mAP50 | Precision |
|---|---|---|
| Prosper Trail | 99.5% | 98%+ |
| First Street | 99.0% | — (retraining in progress) |

- **TRAIN_DETECTED threshold:** 70% confidence
- **Possible train / burst mode:** 20% confidence
- **74+ train events logged** since March 2026

---

## Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv11 (`ultralytics`) — one model per camera |
| Browser automation | Selenium + headless Chrome + Xvfb |
| API | Flask + Flask-CORS (port 5000) |
| Database | SQLite |
| Frontend | Vanilla HTML/CSS/JS — single file, no build step |
| Server | DigitalOcean Ubuntu 22.04, 1 vCPU / 1GB RAM |
| Training | Roboflow (labeling) + Google Colab T4 GPU |

**Monthly cost: ~$19** ($18 DigitalOcean + $1 domain). Cost to the city: $0.

---

## Repository Structure

```
prosper-rail-watch/
├── screen_capture_detector.py   # Main detection + API server
├── simple-rail-watch.html       # Website frontend (single file)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

**Not in this repo** (too large or contains live data):
- `*.pt` — YOLOv11 model weights (kept on server)
- `detections.db` — SQLite database
- `screenshots/` — captured images (45-day retention)
- `training_capture/` — training image sessions

---

## Key Design Decisions

**Why open/close Chrome on every capture?**  
Early versions kept the browser open between captures. After ~30 minutes, ipcamlive's web player delivers a frozen frame — the same static image over and over. Opening and closing Chrome fresh every single capture (v3.4) was the fix that made the system reliable. Do not "optimize" this away.

**Why a Chrome lock?**  
Both camera threads run concurrently. Two simultaneous Chrome instances will crash a 1 vCPU / 1GB RAM server. `chrome_lock = threading.Lock()` serializes all Chrome activity.

**Why screen-scraping instead of direct RTSP?**  
The city cameras support RTSP natively. I'm working on getting read-only stream access — that would drop detection latency from ~53 seconds to under 2 seconds. For now, the public web player is the only available feed.

---

## Scheduling

```
Active hours (7 AM – 7 PM CDT):     20s sleep between captures (~53s full cycle)
Off-peak:                            5 minute sleep
Blackout (7:30 PM – 7:00 AM):       Both cameras sleep
Burst mode (train or possible):      60s interval for 10 minutes on that camera
Post-train capture window:           20 mins of no_train images saved after train event ends
Training capture mode:               Target camera captures as fast as possible (~53s cycle)
```

---

## API Endpoints

All endpoints on `localhost:5000` (served by Flask, proxied by Nginx):

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System running status |
| GET | `/api/detections/recent?limit=N&train_only=true` | Recent detections |
| GET | `/api/train-events?days=30` | Train events with duration |
| GET | `/api/trends` | Trends by day/hour/week |
| GET | `/api/stats?days=7` | Detection statistics |
| GET | `/api/detections/export` | CSV export |
| POST | `/api/training_capture` | Start training image collection |
| GET | `/api/training_capture` | Check training mode status |
| POST | `/api/training_capture/stop` | Stop training mode early |

---

## Deployment

```bash
# Upload from Mac
scp ~/Downloads/screen_capture_detector.py root@SERVER_IP:/var/www/prosper-rail-watch/

# On server — backup and restart
cp screen_capture_detector.py screen_capture_detector.py.bak.vX.X
systemctl restart prosper-rail-watch
sleep 35 && cat /var/log/prosper-autostart.log
# Expected: {"status":"running","success":true}
```

---

## Current Status & Known Issues

- **First Street model broken** — City of Prosper repositioned the physical camera in April 2026 without notice. Old model trained on previous angle now returns null confidence. Collecting new training images using `/api/training_capture`, will retrain in Colab.
- **Dawn/dusk detection gap** — Both models lose confidence in low light before 8 AM and after 6 PM.
- **SMS alerts** — Twilio integration in progress. "Enable Alerts" button on site is temporarily hidden.

---

## Future Work

- [ ] Twilio SMS alerts when train detected
- [ ] RTSP stream access (direct from city cameras — drops latency to ~2s)
- [x] Google Analytics for community usage data
- [ ] Historical data API for city website embed
- [ ] Better dawn/dusk detection with targeted retraining

---

## Key Finding

**Wednesday 3 PM is the highest-risk window.** Historical data shows Wednesdays have the most train events (26 logged), with the longest blockage durations during the 3–5 PM school pickup window.

---

## Contact

Built by Aarush Rudraraju — Prosper High School, Class of 2029  
Questions or collaboration: open an issue or reach out via GitHub.
