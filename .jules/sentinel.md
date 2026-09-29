## 2023-10-10 - [Sentinel] Fixed Critical Vulnerability: Insecure Deserialization in rca_engine.py
**Vulnerability:** The RCA Engine component of the extension used `pickle.load()` on `.pkl` files to load custom failure patterns.
**Learning:** `pickle` is highly insecure for untrusted data since it can lead to arbitrary code execution.
**Prevention:** Avoid `pickle` entirely for file loading. Swapped it for `json.load()` and `.json` format where applicable. Re-instantiated classes from loaded `json` dict data explicitly. Always use data-only serialization formats like JSON, XML, or Protobuf.
