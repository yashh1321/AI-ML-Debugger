## 2024-05-18 - [CRITICAL] Fix insecure deserialization in custom patterns loading
**Vulnerability:** Insecure Deserialization via `pickle.load()` on a workspace file (`.vscode/rca_custom_patterns.pkl`).
**Learning:** Python's `pickle` module should never be used to load files from potentially untrusted or shared workspace locations (like `.vscode/`), as it allows arbitrary code execution.
**Prevention:** Always use safe serialization formats like `json` or `yaml.safe_load` when dealing with user-provided or workspace-local data.
