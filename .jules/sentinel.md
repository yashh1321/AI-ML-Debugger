## 2024-09-30 - Fix Workspace RCE in RCA Engine
**Vulnerability:** The Root Cause Analysis Engine blindly deserialized failure patterns from the workspace using `pickle.load`. A malicious repository containing a `.vscode/failure-patterns.pkl` file could gain arbitrary code execution on the user's machine as soon as they loaded a workspace and the extension activated.
**Learning:** Python's `pickle` module should never be used to deserialize data from untrusted sources, including workspace configurations.
**Prevention:** Replaced `pickle.load` with `json.load` and correctly parse the data safely, enforcing the `.json` extension instead of `.pkl`.
