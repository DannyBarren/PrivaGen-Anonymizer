# Environment variables and Python setup

For **first-time launch**, use **[docs/QUICKSTART.md](docs/QUICKSTART.md)** (Web UI + Anaconda).

For **production secrets and B2**, see **[docs/DEPLOY.md](docs/DEPLOY.md)** and the tables below.

---

## `.env` — API keys (never commit real values)

The root `.env` file lists environment variables the pipeline reads, with **mock placeholders** safe for GitHub. Real secrets are loaded at startup by `scripts/secrets_manager.py` from an **encrypted `.env.enc`** (recommended) or a **permission-locked `.env.local`** — see the full flow in **[docs/DEPLOY.md → Phase 2](docs/DEPLOY.md)**.

Quick start on a private host:

```bash
python -m scripts.secrets_manager gen-key                 # one-time Fernet key (0600)
cp .env.example .env.local && chmod 600 .env.local        # fill real B2 keys
python -m scripts.secrets_manager encrypt --in .env.local --out .env.enc
shred -u .env.local                                       # remove plaintext
python -m scripts.secrets_manager check                   # verify required vars load
```

Loader guarantees: platform env vars are never overwritten, placeholder values are ignored, group/world-readable secret files are refused (`DATASET_ANON_ALLOW_INSECURE_ENV=1` to override), and secret **values are never logged**. `.env.enc`, `secret.key`, and `*.key` are git-ignored.

Secret-loading control variables:

| Variable | Purpose |
|----------|---------|
| `DATASET_ANON_SECRET_KEY` | Fernet key supplied directly (highest priority; avoids on-disk key) |
| `DATASET_ANON_SECRET_KEY_FILE` | Path to a Fernet key file (default `~/.config/dataset_anonymizer/secret.key`) |
| `DATASET_ANON_ALLOW_INSECURE_ENV` | Set `1` to bypass the file-permission check (not recommended) |

This repo is **public** — never commit real keys.

| Variable | Purpose |
|----------|---------|
| `B2_KEY_ID` | Backblaze application key ID |
| `B2_READONLY_KEY` | Read-only key — **ingest only** |
| `B2_WRITE_KEY` | Write key — **export only** |
| `B2_READONLY_BUCKET` / `B2_WRITE_BUCKET` | Bucket names |
| `B2_INGEST_REMOTE_PATH` / `B2_EXPORT_REMOTE_PATH` | rclone path prefixes |
| `RCLONE_BINARY` / `RCLONE_CONFIG` | rclone executable and config file |
| `RCLONE_CRYPT_PASSWORD` | Optional encrypted export |
| `RCLONE_CRYPT_PASSWORD2` | Optional second crypt password |
| `RCLONE_CRYPT_SALT` | Optional crypt salt |
| `OPENAI_API_KEY` | Optional CrewAI QA (`qa.use_crewai_llm: true`) |
| `FLASK_HOST` / `FLASK_PORT` | Web UI bind (default `127.0.0.1:5000`) |

Also set `backblaze.source_bucket` and `backblaze.dest_bucket` in `config.yaml`.

---

## Python environment (Anaconda recommended)

```bash
conda create -n privagen python=3.10 -y
conda activate privagen
```

**UI only (launch dashboard first):**

```bash
pip install -r requirements-ui.txt
python app.py
```

**Full stack (UI install button or CLI):**

```bash
pip install -r requirements.txt
# or
python setup_environment.py
```

Check readiness:

```bash
python -m scripts.environment_checker
python -m scripts.health_check
```

Cached UI status: `reports/ui_environment_status.json`

---

## Related documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md)  
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)  
- [README.md](README.md)
