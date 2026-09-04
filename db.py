"""
db.py — SQLite log of predictions served by the web app.

This is NOT a dataset. The curated dataset stays as files on disk plus
labels.csv, because every image in it has passed a manual review (composition
rules, no vehicles in stable examples, site-leakage checks, licence provenance)
and because a text file gives reviewable diffs in git. Nothing written here ever
reaches slope_dataset/.

The app does not accept image uploads into any queue — there is no public
contribution path. What this records is the *shape* of what the model returns,
with no image kept: one row per analysis holding the probability, the assessment
and the frame size. That supports two things:

  * calibration monitoring — the deployed probe was badly overconfident before
    sigmoid calibration, and drift back toward that is visible without labels as
    most predictions piling into the outermost confidence bins;
  * rate limiting, so one caller cannot monopolise a single-box research demo.

A note on what is NOT stored: no image data, and no IP address in the clear.
Rate limiting needs to recognise a repeat caller, not identify one, so only a
salted hash is kept.
"""

import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Overridable so a deployment can put the file somewhere other than the working
# directory; the default keeps local development writing into the project.
DB_PATH = Path(os.environ.get("GROUNDTRUTH_DB") or "groundtruth.db")

# Salt for the client-address hash. Set GROUNDTRUTH_IP_SALT in the environment
# to make the hashes stable across restarts; the random default means a restart
# resets rate-limit buckets, which is the safe way to fail.
_IP_SALT = os.environ.get("GROUNDTRUTH_IP_SALT") or uuid.uuid4().hex

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    p_unstable   REAL NOT NULL,
    assessment   TEXT,
    sky_cropped  REAL,
    width        INTEGER,
    height       INTEGER,
    client_fp    TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""

# Indexes over columns that arrive via _MIGRATIONS. These cannot live in SCHEMA:
# it runs before the migrations, so on a database created by an earlier version
# the index would reference a column that does not exist yet and the whole
# script would fail.
POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_predictions_client ON predictions(client_fp, created_at);
"""

# Columns added after the first version shipped. SQLite cannot add a column
# conditionally in plain SQL, so they are applied one at a time and skipped if
# already present.
_MIGRATIONS = [
    ("predictions", "client_fp", "TEXT"),
]

# Dropped when the public submission feature was removed. Left here so an
# existing database is cleaned up on open rather than silently retaining
# uploads and submitter contact details that nothing reads any more.
_DROPPED_TABLES = ["submissions"]

_initialised: set[str] = set()


@contextmanager
def connect(db_path: Path | None = None):
    """
    Open a connection, commit on clean exit, roll back on error, always close.

    The close matters: this is called per HTTP request, and `with
    sqlite3.connect(...)` commits the transaction but leaves the handle open.
    Under a long-running server that leaks file handles until the process dies.
    """
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets reads proceed while a write is in flight.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 15000")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init(db_path: Path | None = None) -> None:
    """Create the schema, apply migrations, drop retired tables. Runs once per path."""
    db_path = db_path or DB_PATH
    key = str(db_path)
    if key in _initialised:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table, column, coltype in _MIGRATIONS:
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        conn.executescript(POST_MIGRATION_INDEXES)
        for table in _DROPPED_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    _initialised.add(key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def client_fingerprint(address: str | None) -> str | None:
    """Salted hash of a client address. Never store the address itself."""
    if not address:
        return None
    return hashlib.sha256(f"{_IP_SALT}:{address}".encode()).hexdigest()[:32]


def recent_count(client_fp: str | None, table: str, minutes: int,
                 db_path: Path | None = None) -> int:
    """How many rows this client wrote to `table` in the last `minutes`."""
    if not client_fp:
        return 0
    if table != "predictions":
        raise ValueError(f"unsupported table: {table}")
    init(db_path)
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM predictions WHERE client_fp=? AND created_at>=?",
            (client_fp, since)).fetchone()["c"]


def log_prediction(p_unstable: float, assessment: str, sky_cropped: float,
                   width: int, height: int, client_fp: str | None = None,
                   db_path: Path | None = None) -> None:
    init(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO predictions
               (created_at, p_unstable, assessment, sky_cropped, width, height, client_fp)
               VALUES (?,?,?,?,?,?,?)""",
            (_now(), p_unstable, assessment, sky_cropped, width, height, client_fp),
        )


def counts(db_path: Path | None = None) -> dict:
    init(db_path)
    with connect(db_path) as conn:
        return {"predictions": conn.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]}


def calibration_summary(bins: int = 5, db_path: Path | None = None) -> list[dict]:
    """
    Distribution of served predictions across confidence bands.

    This is a monitoring aid, not an accuracy measure — nothing here knows the
    true label. It answers "is the deployed probe drifting back toward the
    overconfidence that calibration fixed", which is visible without labels:
    a healthy calibrated probe should not put most of its mass in the extreme
    bins.
    """
    init(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT MIN(CAST(p_unstable * {bins} AS INTEGER), {bins - 1}) AS b,
                       COUNT(*) c
                FROM predictions GROUP BY b ORDER BY b""").fetchall()
    width = 1.0 / bins
    return [{"from": r["b"] * width, "to": (r["b"] + 1) * width, "count": r["c"]}
            for r in rows]
