## 2024-05-18 - Missing Database Indexes in Timeline Logs
**Learning:** The `telemetry_timeline.py` file heavily relies on SQLite for logging system metrics and performance events every 500ms. Querying this data (via `get_timeline_data`) requires scanning large tables and sorting by `timestamp_ms` without indexes, causing O(N) performance degradation on large training runs.
**Action:** Always add composite indexes (e.g., `(session_id, timestamp_ms)`) when creating SQLite tables that will be continuously appended to and queried by specific sessions/times.
