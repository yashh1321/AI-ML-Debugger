## 2024-09-30 - Database Indexes added to python backend
**Learning:** The Python backend heavily relies on SQLite databases for logging timeline events, metrics, and profiles, but previously did not use indexes on frequently queried fields like `session_id` and `timestamp_ms`, leading to O(n) lookups.
**Action:** When adding new database tables or extending existing ones in the backend, always include composite or single-column indexes for frequently filtered and ordered columns to ensure O(log n) lookups.
