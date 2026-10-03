## 2024-05-15 - Missing DB Indexes
**Learning:** Found multiple backend scripts querying SQLite databases frequently by `session_id` without any indexing, causing potential O(N) lookup issues during performance monitoring and querying.
**Action:** Add `CREATE INDEX` statements for `session_id` across tables like `performance_events`, `system_metrics`, `layer_timings`, `timeline_markers` etc., in `telemetry_timeline.py` to improve timeline DB performance.
