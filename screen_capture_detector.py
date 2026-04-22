"""
Prosper Rail Watch - Screen Capture Detection
VERSION: 4.9 - Sqrt Bubble Scaling for Detection Timeline
Last Updated: April 22, 2026
Features: SQLite Database, Screenshot Saving, YOLO AI Detection, Smart Scheduling,
          Blackout Hours (7PM-7AM), Burst Mode (60s), Possible Train Folder,
          Train-Only Bounding Boxes, Staggered Camera Starts,
          Historical Trend Fields (day_of_week, hour_of_day, month, week_number),
          Train-Only API Filter (?train_only=true), 45-day screenshot retention,
          Training Capture Mode (POST /api/training_capture)
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import numpy as np
from PIL import Image
import io
import cv2
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
from datetime import datetime
import json
import pytz
import sqlite3
from contextlib import contextmanager

def is_within_monitoring_hours():
    """Check if current time is within configured monitoring windows"""
    if not MONITORING_SCHEDULE['enabled']:
        return True
    tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
    now = datetime.now(tz)
    current_time = now.strftime('%H:%M')
    for window in MONITORING_SCHEDULE['time_windows']:
        if window['start'] <= current_time <= window['end']:
            return True
    return False


def is_blackout_hours():
    """Check if current time is within the nighttime blackout window"""
    blackout = MONITORING_SCHEDULE.get('blackout', {})
    if not blackout.get('enabled', False):
        return False
    tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
    now = datetime.now(tz)
    current_time = now.strftime('%H:%M')
    start = blackout['start']
    end = blackout['end']
    if start > end:
        return current_time >= start or current_time < end
    else:
        return start <= current_time < end


def seconds_until_blackout_ends():
    """Calculate seconds until the blackout window ends"""
    blackout = MONITORING_SCHEDULE.get('blackout', {})
    tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
    now = datetime.now(tz)
    current_minutes = now.hour * 60 + now.minute
    end_h, end_m = map(int, blackout['end'].split(':'))
    end_minutes = end_h * 60 + end_m
    diff_minutes = end_minutes - current_minutes
    if diff_minutes <= 0:
        diff_minutes += 24 * 60
    return int(diff_minutes * 60)


def seconds_until_next_window():
    """Returns seconds until next monitoring window, max 300 (5 min)."""
    tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
    now = datetime.now(tz)
    current_minutes = now.hour * 60 + now.minute
    min_wait = 300
    for window in MONITORING_SCHEDULE['time_windows']:
        start_h, start_m = map(int, window['start'].split(':'))
        window_start_minutes = start_h * 60 + start_m
        diff_minutes = window_start_minutes - current_minutes
        if diff_minutes <= 0:
            diff_minutes += 24 * 60
        diff_seconds = diff_minutes * 60
        min_wait = min(min_wait, diff_seconds)
    return int(min_wait)


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_CONFIG['db_path'])
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        raise
    finally:
        conn.close()


def init_database():
    """Initialize the database with required tables"""
    if not DB_CONFIG['enabled']:
        return
    try:
        with get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    camera_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    train_detected BOOLEAN NOT NULL,
                    confidence REAL,
                    detection_details TEXT,
                    screenshot_path TEXT,
                    is_school_day BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            try:
                conn.execute('ALTER TABLE detections ADD COLUMN is_school_day BOOLEAN DEFAULT 0')
            except Exception:
                pass
            conn.execute('''
                CREATE TABLE IF NOT EXISTS train_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    camera_name TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    duration_seconds INTEGER,
                    is_school_day BOOLEAN DEFAULT 0,
                    peak_hours BOOLEAN DEFAULT 0,
                    max_confidence REAL,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    month INTEGER,
                    week_number INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            for col in ['day_of_week INTEGER', 'hour_of_day INTEGER', 'month INTEGER', 'week_number INTEGER']:
                try:
                    conn.execute(f'ALTER TABLE train_events ADD COLUMN {col}')
                except Exception:
                    pass
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_camera_timestamp ON detections(camera_id, timestamp DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_train_events_camera ON train_events(camera_id, start_time DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_train_detected ON detections(train_detected, timestamp DESC)')
            print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")


def check_is_school_day():
    """Returns True if today is a weekday (Mon-Fri). Does not account for holidays."""
    tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
    now = datetime.now(tz)
    return now.weekday() < 5


def save_detection_to_db(camera_id, camera_name, train_detected, confidence,
                         detections_info, screenshot_path):
    """Save detection record to database"""
    if not DB_CONFIG['enabled']:
        return
    try:
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO detections
                (camera_id, camera_name, timestamp, train_detected,
                 confidence, detection_details, screenshot_path, is_school_day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_id,
                camera_name,
                datetime.now(),
                train_detected,
                confidence,
                json.dumps(detections_info) if detections_info else None,
                screenshot_path,
                check_is_school_day()
            ))
            print(f"💾 Detection saved to database (train_detected={train_detected}, school_day={check_is_school_day()})")
    except Exception as e:
        print(f"❌ Failed to save to database: {e}")


def get_recent_detections(limit=100, camera_id=None, train_only=False):
    """
    Retrieve recent detections from database.
    train_only=True  → only confirmed train detections (train_detected=1) — used by history tab
    train_only=False → all detections — used by live status polling
    """
    if not DB_CONFIG['enabled']:
        return []
    try:
        with get_db_connection() as conn:
            if camera_id and train_only:
                cursor = conn.execute('''
                    SELECT * FROM detections
                    WHERE camera_id = ? AND train_detected = 1
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (camera_id, limit))
            elif camera_id:
                cursor = conn.execute('''
                    SELECT * FROM detections
                    WHERE camera_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (camera_id, limit))
            elif train_only:
                cursor = conn.execute('''
                    SELECT * FROM detections
                    WHERE train_detected = 1
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            else:
                cursor = conn.execute('''
                    SELECT * FROM detections
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"❌ Failed to retrieve detections: {e}")
        return []


def get_detection_stats(days=7):
    """Get detection statistics for the past N days"""
    if not DB_CONFIG['enabled']:
        return {}
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT
                    camera_name,
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN train_detected = 1 THEN 1 ELSE 0 END) as train_detections,
                    AVG(CASE WHEN train_detected = 1 THEN confidence ELSE NULL END) as avg_confidence
                FROM detections
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY camera_name
            ''', (days,))
            stats = {}
            for row in cursor.fetchall():
                stats[row['camera_name']] = {
                    'total_checks': row['total_checks'],
                    'train_detections': row['train_detections'],
                    'avg_confidence': round(row['avg_confidence'], 1) if row['avg_confidence'] else 0,
                    'detection_rate': round((row['train_detections'] / row['total_checks'] * 100), 1) if row['total_checks'] > 0 else 0
                }
            return stats
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return {}


def save_train_event(camera_id, camera_name, start_time, end_time, max_confidence):
    """Save a complete train event with duration and trend fields to database"""
    if not DB_CONFIG['enabled']:
        return
    try:
        tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
        start_local = start_time.astimezone(tz) if start_time.tzinfo else tz.localize(start_time)
        end_aware = end_time.astimezone(tz) if end_time.tzinfo else tz.localize(end_time)
        # Conservative estimate: gates close ~60s before train arrives, and train
        # could have left up to one burst interval (60s) after last detection.
        measured_seconds = int((end_aware - start_local).total_seconds())
        duration_seconds = measured_seconds + 120
        start_time_str = start_local.strftime('%H:%M')
        peak_hours = any(
            w['start'] <= start_time_str <= w['end']
            for w in MONITORING_SCHEDULE['time_windows']
        )
        day_of_week = int(start_local.strftime('%w'))
        hour_of_day = start_local.hour
        month = start_local.month
        week_number = int(start_local.strftime('%W'))
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO train_events
                (camera_id, camera_name, start_time, end_time, duration_seconds,
                 is_school_day, peak_hours, max_confidence,
                 day_of_week, hour_of_day, month, week_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_id, camera_name, start_time, end_time, duration_seconds,
                check_is_school_day(), peak_hours, max_confidence,
                day_of_week, hour_of_day, month, week_number
            ))
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        print(f"🚂 Train event saved: {camera_name} | Duration: {duration_seconds}s ({duration_seconds//60}m {duration_seconds%60}s) | "
              f"School day: {check_is_school_day()} | Peak: {peak_hours} | "
              f"{day_names[day_of_week]} {hour_of_day:02d}:00 | Week {week_number} | Month {month}")
    except Exception as e:
        print(f"❌ Failed to save train event: {e}")


app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────

CAMERA_URLS = {
    1: {
        'name': 'First Street',
        'url': 'https://g1.ipcamlive.com/player/player.php?alias=firstst&autoplay=1',
        'location': 'First St, Prosper, TX'
    },
    2: {
        'name': 'Prosper Trail',
        'url': 'https://g1.ipcamlive.com/player/player.php?alias=prospertrl&autoplay=1',
        'location': 'Prosper Trail, Prosper, TX'
    }
}

MONITORING_SCHEDULE = {
    'enabled': True,
    'time_windows': [
        {'start': '07:00', 'end': '19:00'},
    ],
    'blackout': {
        'enabled': True,
        'start': '19:00',
        'end': '07:00'
    },
    'timezone': 'America/Chicago'
}

SCREENSHOT_CONFIG = {
    'enabled': True,
    'save_path': '/var/www/prosper-rail-watch/screenshots',
    'save_all_frames': False,
    'draw_boxes': True,
    'organize_by_result': True,
    'keep_days': 45
}

BURST_CONFIG = {
    'enabled': True,
    'possible_train_confidence': 20,
    'burst_confidence': 70,
    'burst_interval_seconds': 60,
    'burst_duration_seconds': 600,
}

POST_TRAIN_SAVE_MINUTES = 20

DB_CONFIG = {
    'enabled': True,
    'db_path': '/var/www/prosper-rail-watch/detections.db'
}

detections = []
system_running = False
browsers = {}
chrome_lock = threading.Lock()

# ── Training capture mode ──────────────────────────────────────────────────────
TRAINING_CAPTURE_DIR = "/var/www/prosper-rail-watch/training_capture"

training_mode = {
    'active':      False,
    'end_time':    None,
    'image_count': 0,
    'session_dir': None,
    'camera_id':   1,     # Default: First Street
    'interval':    5,     # Seconds between training captures
}
training_lock = threading.Lock()


# ── Detector class ─────────────────────────────────────────────────────────────

class ScreenCaptureDetector:
    """Captures screenshots and detects trains"""

    def __init__(self, camera_id, camera_info):
        self.camera_id = camera_id
        self.camera_info = camera_info
        self.driver = None
        self.is_running = False
        self.last_detection_time = 0
        self.cooldown_seconds = 60
        self.model = self._load_model()
        if SCREENSHOT_CONFIG['enabled']:
            self.setup_screenshot_directories()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = '/var/www/prosper-rail-watch/first_street_best.pt' if self.camera_id == 1 \
                         else '/var/www/prosper-rail-watch/prosper_trail_best.pt'
            model = YOLO(model_path)
            print(f"✅ [{self.camera_info['name']}] AI model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"⚠️  [{self.camera_info['name']}] Failed to load model: {e}")
            return None

    def setup_screenshot_directories(self):
        base_path = SCREENSHOT_CONFIG['save_path']
        camera_name = self.camera_info['name'].replace(' ', '_').lower()
        if SCREENSHOT_CONFIG['organize_by_result']:
            self.screenshot_dirs = {
                'trains': os.path.join(base_path, camera_name, 'trains_detected'),
                'no_trains': os.path.join(base_path, camera_name, 'no_trains'),
                'possible_trains': os.path.join(base_path, camera_name, 'possible_trains')
            }
        else:
            self.screenshot_dirs = {'all': os.path.join(base_path, camera_name, 'all_frames')}
        for dir_path in self.screenshot_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Screenshot directories created for {self.camera_info['name']}")

    def save_screenshot_with_detections(self, frame, detections_info, train_detected):
        if not SCREENSHOT_CONFIG['enabled']:
            return None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if SCREENSHOT_CONFIG['draw_boxes'] and detections_info:
                annotated_frame = frame.copy()
                for detection in detections_info:
                    if detection['class'].lower() != 'train':
                        continue
                    x1, y1, x2, y2 = detection['box']
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(annotated_frame, f"TRAIN: {detection['confidence']:.1f}%",
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                annotated_frame = frame
            if SCREENSHOT_CONFIG['organize_by_result']:
                if train_detected:
                    save_dir = self.screenshot_dirs['trains']
                elif 'possible_trains' in self.screenshot_dirs and \
                     any(d['class'].lower() == 'train' and d['confidence'] >= BURST_CONFIG['possible_train_confidence']
                         for d in (detections_info or [])):
                    save_dir = self.screenshot_dirs['possible_trains']
                else:
                    save_dir = self.screenshot_dirs['no_trains']
            else:
                save_dir = self.screenshot_dirs['all']
            filename = f"{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'camera': self.camera_info['name'],
                'train_detected': train_detected,
                'detections': detections_info
            }
            with open(os.path.join(save_dir, f"{timestamp}.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Screenshot saved: {filename}")
            return filepath
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None

    def capture_frame(self):
        """Open browser, capture frame, close immediately. Chrome lock prevents concurrent opens."""
        driver = None
        print(f"⏳ [{self.camera_info['name']}] Waiting for Chrome lock...")
        with chrome_lock:
            print(f"🔓 [{self.camera_info['name']}] Chrome lock acquired - opening browser")
            try:
                chrome_options = Options()
                chrome_options.binary_location = '/usr/bin/google-chrome-stable'
                chrome_options.add_argument('--headless=new')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                chrome_options.add_argument('--mute-audio')
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                driver.set_page_load_timeout(30)
                start_time = time.time()
                try:
                    driver.get(self.camera_info['url'])
                except Exception:
                    print(f"⚠️  [{self.camera_info['name']}] Page load timed out after 30s - skipping capture")
                    return None
                elapsed = time.time() - start_time
                time.sleep(max(0, 10 - elapsed))
                screenshot = driver.get_screenshot_as_png()
                image = Image.open(io.BytesIO(screenshot))
                frame = np.array(image)
                print(f"✓ [{self.camera_info['name']}] Frame captured in {time.time()-start_time:.1f}s")
                return frame
            except Exception as e:
                print(f"✗ [{self.camera_info['name']}] Error capturing frame: {e}")
                return None
            finally:
                if driver:
                    try:
                        driver.quit()
                        print(f"🔒 [{self.camera_info['name']}] Browser closed, lock released")
                    except Exception:
                        pass

    def detect_train_simple(self, frame):
        """Fallback simple detection (placeholder)"""
        confidence = 0.0
        if np.random.random() > 0.95:
            confidence = 0.85 + np.random.random() * 0.14
        return confidence > 0.7

    def detect_train_ai(self, frame):
        """AI-based train detection using custom YOLO models per camera"""
        try:
            if self.model is None:
                raise ImportError("Model not loaded")
            results = self.model(frame, conf=0.3, verbose=False)
            detections_info = []
            train_detected = False
            print(f"🔍 YOLO Detection Results:")
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    print(f"   ⭕ No objects detected in frame")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0]) * 100
                    class_name = result.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detection_data = {'class': class_name, 'class_id': class_id, 'confidence': confidence, 'box': [x1, y1, x2, y2]}
                    detections_info.append(detection_data)
                    emoji = "🚂" if class_name.lower() == 'train' else "🚗" if class_name.lower() == 'car' else "📦"
                    print(f"   {emoji} Detected: {class_name} (confidence: {confidence:.1f}%)")
                    # Custom model uses 'Train' (capital T) — always use .lower()
                    if class_name.lower() == 'train' and confidence >= 70:
                        train_detected = True
            if train_detected:
                train_dets = [d for d in detections_info if d['class'].lower() == 'train']
                print(f"   ✅ TRAIN DETECTED! Highest confidence: {max(d['confidence'] for d in train_dets):.1f}%")
            else:
                train_confs = [d['confidence'] for d in detections_info if d['class'].lower() == 'train']
                if train_confs:
                    print(f"   ⚠️  Train below threshold: {max(train_confs):.1f}% (threshold: 70%)")
                else:
                    print(f"   ⭕ No trains detected (total objects: {len(detections_info)})")
            return train_detected, detections_info
        except ImportError:
            print(f"   ⚠️  YOLO not available, using fallback detection")
            return self.detect_train_simple(frame), []
        except Exception as e:
            print(f"   ❌ AI detection error: {e}")
            return self.detect_train_simple(frame), []

    def monitor(self, stagger_seconds=0):
        """Main monitoring loop with burst mode"""
        if stagger_seconds:
            print(f"⏱️  [{self.camera_info['name']}] Staggering start by {stagger_seconds}s...")
            time.sleep(stagger_seconds)
        print(f"🚂 Monitoring {self.camera_info['name']}...")
        self.is_running = True
        frame_count = 0
        burst_mode = False
        burst_end_time = 0
        train_event_active = False
        train_event_start = None
        train_event_max_conf = 0
        post_train_capture_until = None

        while self.is_running:
            try:
                # ── Training capture mode check ────────────────────────────
                with training_lock:
                    t_active    = training_mode['active']
                    t_camera_id = training_mode['camera_id']
                    t_end_time  = training_mode['end_time']
                    t_dir       = training_mode['session_dir']

                if t_active:
                    # Auto-expire when duration is up
                    if time.time() >= t_end_time:
                        with training_lock:
                            training_mode['active'] = False
                        print(f"🏁 [TRAINING] Session complete — {training_mode['image_count']} images saved to {t_dir}")
                        print(f"✅ [TRAINING] Returning to normal monitoring")
                        continue

                    # Only the target camera captures during training mode
                    # Prosper Trail sleeps 20s between checks so it's not
                    # spinning, but picks up quickly when training ends
                    if self.camera_id != t_camera_id:
                        time.sleep(20)
                        continue

                    # Capture frame using existing mechanism (chrome_lock respected)
                    print(f"📸 [TRAINING] Capturing image for training...")
                    frame = self.capture_frame()
                    if frame is not None:
                        os.makedirs(t_dir, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        with training_lock:
                            training_mode['image_count'] += 1
                            count = training_mode['image_count']
                        filename = f"first_street_train_{timestamp}_{count:04d}.jpg"
                        filepath = os.path.join(t_dir, filename)
                        cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        print(f"💾 [TRAINING] [{count}] Saved: {filename}")
                    else:
                        print(f"⚠️  [TRAINING] Frame capture failed — retrying in 20s")

                    # 20s recovery buffer — same as normal peak monitoring
                    # keeps CPU from spiking and allows recovery if capture runs long
                    time.sleep(20)
                    continue
                # ── End training mode check ────────────────────────────────

                if is_blackout_hours() and not burst_mode:
                    wait_seconds = seconds_until_blackout_ends()
                    print(f"🌙 [{self.camera_info['name']}] Blackout hours. Sleeping {round(wait_seconds/3600,1)}hrs until 7:00 AM...")
                    time.sleep(wait_seconds)
                    continue
                elif is_blackout_hours() and burst_mode:
                    print(f"🌙 [{self.camera_info['name']}] Blackout but burst active — completing burst first...")

                if burst_mode and time.time() >= burst_end_time:
                    burst_mode = False
                    print(f"🏁 [{self.camera_info['name']}] Burst ended. Returning to normal 20s interval.")

                if burst_mode:
                    interval = BURST_CONFIG['burst_interval_seconds']
                    mode_label = '[BURST MODE] '
                else:
                    interval = 20
                    mode_label = ''

                print(f"🎬 [{self.camera_info['name']}] {mode_label}About to capture frame...")
                frame = self.capture_frame()
                print(f"✓ [{self.camera_info['name']}] Frame capture completed")

                if frame is not None:
                    frame_count += 1
                    print(f"📸 [{self.camera_info['name']}] Frame {frame_count} captured at {datetime.now().strftime('%H:%M:%S')}")
                    print(f"🔍 [{self.camera_info['name']}] Running AI detection...")
                    train_detected, detections_info = self.detect_train_ai(frame)
                    train_confidences = [d['confidence'] for d in detections_info if d['class'].lower() == 'train']
                    max_train_conf = max(train_confidences) if train_confidences else 0
                    screenshot_path = None
                    if SCREENSHOT_CONFIG['enabled']:
                        in_post_train_window = post_train_capture_until is not None and time.time() < post_train_capture_until
                        if train_detected or max_train_conf >= BURST_CONFIG['possible_train_confidence'] or in_post_train_window:
                            if not train_detected and max_train_conf < BURST_CONFIG['possible_train_confidence'] and in_post_train_window:
                                remaining = (post_train_capture_until - time.time()) / 60
                                print(f"📸 [{self.camera_info['name']}] Post-train no_train saved ({remaining:.1f} mins remaining)")
                            screenshot_path = self.save_screenshot_with_detections(frame, detections_info, train_detected)
                    confidence = max_train_conf if max_train_conf > 0 else None
                    save_detection_to_db(
                        camera_id=self.camera_id, camera_name=self.camera_info['name'],
                        train_detected=train_detected, confidence=confidence,
                        detections_info=detections_info, screenshot_path=screenshot_path
                    )

                    if train_detected:
                        print(f"✅ [{self.camera_info['name']}] Train detected! Confidence: {max_train_conf:.1f}%")
                        current_time = time.time()
                        if not train_event_active:
                            train_event_active = True
                            _tz = pytz.timezone(MONITORING_SCHEDULE['timezone'])
                            train_event_start = datetime.now(_tz)
                            train_event_max_conf = max_train_conf
                            print(f"🕐 [{self.camera_info['name']}] Train event started at {train_event_start.strftime('%H:%M:%S')}")
                        else:
                            train_event_max_conf = max(train_event_max_conf, max_train_conf)
                        if BURST_CONFIG['enabled'] and not burst_mode:
                            burst_mode = True
                            burst_end_time = current_time + BURST_CONFIG['burst_duration_seconds']
                            burst_stagger = 15 if self.camera_id == 2 else 0
                            if burst_stagger:
                                print(f"🚨 [{self.camera_info['name']}] BURST MODE ACTIVATED! Staggering {burst_stagger}s...")
                                time.sleep(burst_stagger)
                            else:
                                print(f"🚨 [{self.camera_info['name']}] BURST MODE ACTIVATED! Every {BURST_CONFIG['burst_interval_seconds']}s for {BURST_CONFIG['burst_duration_seconds']//60} min")
                        if current_time - self.last_detection_time >= self.cooldown_seconds:
                            print(f"🚂 TRAIN DETECTED at {self.camera_info['name']}!")
                            train_confs = [d['confidence'] for d in detections_info if d['class'].lower() == 'train']
                            detection = {
                                'camera_id': self.camera_id, 'camera_name': self.camera_info['name'],
                                'timestamp': datetime.now().isoformat(),
                                'confidence': max(train_confs) if train_confs else 0.95,
                                'method': 'screen_capture', 'detections': detections_info
                            }
                            detections.append(detection)
                            self.last_detection_time = current_time

                    elif max_train_conf >= BURST_CONFIG['possible_train_confidence']:
                        print(f"🔶 [{self.camera_info['name']}] Possible train ({max_train_conf:.1f}%) - saved to possible_trains/")
                        post_train_capture_until = time.time() + POST_TRAIN_SAVE_MINUTES * 60
                        print(f"📸 [{self.camera_info['name']}] Post-train capture window started ({POST_TRAIN_SAVE_MINUTES} mins)")
                        if train_event_active:
                            save_train_event(self.camera_id, self.camera_info['name'],
                                             train_event_start, datetime.now(), train_event_max_conf)
                            train_event_active = False; train_event_start = None; train_event_max_conf = 0
                        if BURST_CONFIG['enabled'] and not burst_mode:
                            burst_mode = True
                            burst_end_time = time.time() + BURST_CONFIG['burst_duration_seconds']
                            burst_stagger = 15 if self.camera_id == 2 else 0
                            if burst_stagger:
                                print(f"🚨 [{self.camera_info['name']}] BURST MODE ACTIVATED on possible train! Staggering {burst_stagger}s...")
                                time.sleep(burst_stagger)
                            else:
                                print(f"🚨 [{self.camera_info['name']}] BURST MODE ACTIVATED on possible train!")
                    else:
                        print(f"⭕ [{self.camera_info['name']}] No train detected - crossing clear")
                        if train_event_active:
                            save_train_event(self.camera_id, self.camera_info['name'],
                                             train_event_start, datetime.now(), train_event_max_conf)
                            train_event_active = False; train_event_start = None; train_event_max_conf = 0
                            post_train_capture_until = time.time() + POST_TRAIN_SAVE_MINUTES * 60
                            print(f"📸 [{self.camera_info['name']}] Post-train capture window started ({POST_TRAIN_SAVE_MINUTES} mins)")

                print(f"⏰ [{self.camera_info['name']}] {mode_label}Waiting {interval}s until next check...")
                time.sleep(interval)

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def stop(self):
        self.is_running = False


# ── Monitoring control ─────────────────────────────────────────────────────────

def start_monitoring():
    global system_running, browsers
    if system_running:
        return
    system_running = True
    print("\n🚀 Starting screen capture detection system...")
    for index, (camera_id, camera_info) in enumerate(CAMERA_URLS.items()):
        detector = ScreenCaptureDetector(camera_id, camera_info)
        browsers[camera_id] = detector
        stagger = index * 30
        thread = threading.Thread(target=detector.monitor, kwargs={'stagger_seconds': stagger}, daemon=True)
        thread.start()
    print("✓ All cameras monitoring started")


def stop_monitoring():
    global system_running, browsers
    system_running = False
    for detector in browsers.values():
        detector.stop()
    browsers.clear()
    print("✓ Monitoring stopped")


# ── Flask API endpoints ────────────────────────────────────────────────────────

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': 'running' if system_running else 'stopped',
        'version': '4.9',
        'cameras': [
            {'camera_id': cam_id, 'name': info['name'], 'location': info['location'], 'is_active': system_running}
            for cam_id, info in CAMERA_URLS.items()
        ],
        'total_detections': len(detections),
        'method': 'screen_capture'
    })


@app.route('/api/detections')
def get_detections():
    limit = request.args.get('limit', 50, type=int)
    return jsonify({'detections': detections[-limit:][::-1]})


@app.route('/api/start', methods=['POST', 'GET'])
def start():
    start_monitoring()
    return jsonify({'success': True, 'status': 'running'})


@app.route('/api/stop', methods=['POST', 'GET'])
def stop():
    stop_monitoring()
    return jsonify({'success': True, 'status': 'stopped'})


@app.route('/api/test-detection', methods=['POST', 'GET'])
def test_detection():
    if not system_running:
        return jsonify({'success': False, 'message': 'System not running'})
    detection = {
        'camera_id': 1, 'camera_name': 'First Street',
        'timestamp': datetime.now().isoformat(), 'confidence': 0.95,
        'method': 'test', 'test': True
    }
    detections.append(detection)
    return jsonify({'success': True, 'detection': detection})


@app.route('/api/detections/recent', methods=['GET'])
def get_recent_detections_api():
    """
    Get recent detections from database.

    Query params:
      limit      - number of records (default 100)
      camera_id  - filter by camera (optional)
      train_only - 'true' returns only train detections; 'false' returns all (default)

    Usage:
      /api/detections/recent?limit=20              -> last 20 checks (live status polling)
      /api/detections/recent?limit=20&train_only=true -> last 20 train detections (history tab)
    """
    limit = request.args.get('limit', 100, type=int)
    camera_id = request.args.get('camera_id', type=int)
    train_only = request.args.get('train_only', 'false').lower() == 'true'
    db_detections = get_recent_detections(limit=limit, camera_id=camera_id, train_only=train_only)
    return jsonify({
        'success': True,
        'count': len(db_detections),
        'train_only': train_only,
        'detections': db_detections
    })


@app.route('/api/stats', methods=['GET'])
def get_stats_api():
    days = request.args.get('days', 7, type=int)
    stats = get_detection_stats(days=days)
    return jsonify({'success': True, 'period_days': days, 'stats': stats})


@app.route('/api/train-events', methods=['GET'])
def get_train_events():
    days = request.args.get('days', 30, type=int)
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT camera_name, start_time, end_time, duration_seconds,
                       is_school_day, peak_hours, max_confidence,
                       day_of_week, hour_of_day, month, week_number
                FROM train_events
                WHERE start_time >= datetime('now', '-' || ? || ' days')
                ORDER BY start_time DESC
            ''', (days,))
            events = [dict(row) for row in cursor.fetchall()]
        return jsonify({'success': True, 'count': len(events), 'events': events})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/trends', methods=['GET'])
def get_trends():
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT day_of_week, COUNT(*) as count, AVG(duration_seconds) as avg_duration
                FROM train_events WHERE day_of_week IS NOT NULL
                GROUP BY day_of_week ORDER BY day_of_week
            ''')
            by_day = [dict(row) for row in cursor.fetchall()]

            cursor = conn.execute('''
                SELECT hour_of_day, COUNT(*) as count, AVG(duration_seconds) as avg_duration
                FROM train_events WHERE hour_of_day IS NOT NULL
                GROUP BY hour_of_day ORDER BY hour_of_day
            ''')
            by_hour = [dict(row) for row in cursor.fetchall()]

            # Group by week_number ONLY — avoids splitting weeks that span two months
            cursor = conn.execute('''
                SELECT week_number, COUNT(*) as count,
                       MIN(DATE(start_time)) as week_start, MAX(DATE(start_time)) as week_end
                FROM train_events WHERE week_number IS NOT NULL
                GROUP BY week_number ORDER BY week_number
            ''')
            by_week = [dict(row) for row in cursor.fetchall()]

        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        for row in by_day:
            row['day_name'] = day_names[row['day_of_week']]

        return jsonify({'success': True, 'by_day_of_week': by_day, 'by_hour_of_day': by_hour, 'by_week': by_week})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/detections/export', methods=['GET'])
def export_detections():
    import csv
    from io import StringIO
    from flask import Response
    db_detections = get_recent_detections(limit=10000)
    output = StringIO()
    if db_detections:
        fieldnames = ['id', 'camera_name', 'timestamp', 'train_detected', 'confidence', 'screenshot_path']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for detection in db_detections:
            writer.writerow({
                'id': detection['id'], 'camera_name': detection['camera_name'],
                'timestamp': detection['timestamp'], 'train_detected': detection['train_detected'],
                'confidence': detection['confidence'], 'screenshot_path': detection.get('screenshot_path', '')
            })
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=detections.csv'})


@app.route('/api/training_capture', methods=['POST'])
def start_training_capture():
    """
    Trigger training image capture mode on a camera.
    Pauses normal monitoring on that camera, captures images fast, auto-reverts after duration.

    Usage:
      curl -X POST "http://localhost:5000/api/training_capture"
      curl -X POST "http://localhost:5000/api/training_capture?duration=300&camera=1&interval=5"

    Params:
      duration  — seconds to run (default: 900 = 15 min, ~17 images)
      camera    — camera_id (default: 1 = First Street)
    """
    duration  = int(request.args.get('duration', 900))
    camera_id = int(request.args.get('camera', 1))

    camera_name = CAMERA_URLS.get(camera_id, {}).get('name', f'Camera {camera_id}')
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(TRAINING_CAPTURE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    with training_lock:
        training_mode['active']      = True
        training_mode['end_time']    = time.time() + duration
        training_mode['image_count'] = 0
        training_mode['session_dir'] = session_dir
        training_mode['camera_id']   = camera_id

    print(f"🎯 [TRAINING] Started — camera={camera_name}, duration={duration}s (~{duration//53} images expected)")
    print(f"📁 [TRAINING] Saving to: {session_dir}")

    return jsonify({
        'success':        True,
        'status':         'training_active',
        'camera':         camera_name,
        'duration_s':     duration,
        'duration_min':   duration // 60,
        'images_expected': duration // 53,
        'session_dir':    session_dir,
        'tip':            f'scp -r root@157.245.216.46:{session_dir} ~/Desktop/first_street_training/'
    })


@app.route('/api/training_capture', methods=['GET'])
def training_capture_status():
    """Check training mode status — curl http://localhost:5000/api/training_capture"""
    with training_lock:
        if not training_mode['active']:
            return jsonify({'active': False, 'message': 'No training session running'})
        remaining = max(0, int(training_mode['end_time'] - time.time()))
        return jsonify({
            'active':       True,
            'remaining_s':  remaining,
            'image_count':  training_mode['image_count'],
            'session_dir':  training_mode['session_dir'],
            'camera_id':    training_mode['camera_id'],
        })


@app.route('/api/training_capture/stop', methods=['POST'])
def stop_training_capture():
    """Stop training mode early — curl -X POST http://localhost:5000/api/training_capture/stop"""
    with training_lock:
        if not training_mode['active']:
            return jsonify({'success': False, 'message': 'No training session running'})
        training_mode['active'] = False
        count = training_mode['image_count']
        session_dir = training_mode['session_dir']
    print(f"⏹️  [TRAINING] Stopped early — {count} images saved")
    return jsonify({'success': True, 'images_saved': count, 'session_dir': session_dir})


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("PROSPER RAIL WATCH - Screen Capture Detection v4.9")
    print("=" * 70)
    print()
    print("🗄️  Initializing database...")
    init_database()
    print()
    print("🌐 Starting API server on http://localhost:5000")
    print()
    print("📡 Endpoints:")
    print("   - GET  /api/status")
    print("   - GET  /api/detections/recent")
    print("   - GET  /api/detections/recent?train_only=true")
    print("   - GET  /api/train-events")
    print("   - GET  /api/trends")
    print("   - GET  /api/stats")
    print("   - GET  /api/detections/export")
    print("   - POST /api/start")
    print("   - POST /api/stop")
    print("   - POST /api/training_capture        (start training capture)")
    print("   - GET  /api/training_capture        (check training status)")
    print("   - POST /api/training_capture/stop   (stop training early)")
    print()
    print("💡 Start monitoring: curl http://localhost:5000/api/start")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        stop_monitoring()
        print("✓ System stopped")
