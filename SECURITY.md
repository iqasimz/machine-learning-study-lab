# Security

This is a local educational app, not an authenticated public training service.

- Bind only to `127.0.0.1`. Keep its Host/Origin checks. Do not expose it using a reverse proxy, tunnel, port forwarding or a public notebook endpoint.
- A training worker can consume substantial memory, disk, battery and compute. Begin with small settings and stop runs using the dashboard or Ctrl-C. Auto device selection is not a memory-capacity guarantee.
- Downloads happen only through explicit preparation commands and allowlisted sources. Keep pinned revisions, integrity checks and `trust_remote_code=False` behavior. No API key is required.
- Joblib/pickle files can execute code when loaded. Never load files from an unknown source. Do not add an arbitrary model/upload/command endpoint without a new threat review.
- Do not upload private learner datasets, API keys, local run folders or model caches to GitHub. `.gitignore` is a convenience, not a guarantee.
- Report a vulnerability privately using GitHub private vulnerability reporting if enabled. Otherwise contact the repository owner before publishing exploit details. Never put credentials into an issue.

The static Pages preview contains no remote execution backend. Browser progress is stored locally at that website address and can be lost when storage is cleared.
