## 2024-05-18 - [HIGH] Fix SQL Injection in performance_profiler.py
**Vulnerability:** SQL Injection in `performance_profiler.py` using string formatting (`.format()`) for the `hours` parameter in `get_performance_history` and `get_timeline_events`.
**Learning:** Python type hints (like `hours: int`) do not prevent runtime injection of malicious string payloads into SQL queries. String concatenation or formatting for SQL values is always unsafe, even if the parameter appears typed.
**Prevention:** Always use parameterized queries (e.g., `?` in SQLite) for incorporating variables into SQL statements, ensuring the database driver handles escaping and prevents code execution.
