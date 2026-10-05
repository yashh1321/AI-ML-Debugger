## 2024-05-24 - Missing SQLite Indexes in Telemetry Tracking
**Learning:** Found that heavy analytical tables in the `performance_profiler.py` and `telemetry_timeline.py` scripts lacked database indexes on heavily queried columns (`timestamp` and `session_id`). This resulted in O(N) full table scans during data retrieval, which is a significant bottleneck for an extension that logs metrics frequently.
**Action:** Always verify database indexes are present when creating SQLite schemas for append-heavy telemetry tools, particularly on columns used for time-range filtering and session lookups.
