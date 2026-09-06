# Hugging Face Integration — Architecture & Usage Guide

## Package Structure

```
researchforge/adapters/huggingface/
├── __init__.py        # Public API exports
├── errors.py          # Typed error hierarchy
├── client.py          # HTTP client + HFClientConfig
├── inference.py       # Text generation
├── embeddings.py      # Feature extraction + local fallback
├── models.py          # Model discovery via HF Hub API
├── datasets.py        # Dataset discovery via HF Hub API
└── provenance.py      # Cryptographic provenance records
```

## Key Design Decisions

### 1. Offline-First
All providers work without `HF_TOKEN`. Live tests gated on `RESEARCHFORGE_HF_LIVE_TEST=1`.

### 2. Credential Safety
- `HF_TOKEN` read from env at call time, never stored
- Token never appears in error messages, provenance, or artifacts

### 3. Full Provenance
Every provider interaction returns `HFProvenance` with model_id, revision, latency, fingerprints.

### 4. No Automatic Fallback
When HF fails, the pipeline fails deterministically without silently falling back to secondary providers.

### 5. ECRM Compatibility
Default ECRM uses local HashingVectorizer (256-d). HF embeddings are opt-in only.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HF API token | (none, offline) |
| `RF_HF_MODEL_ID` | Model override | `mistralai/Mistral-7B-Instruct-v0.3` |
| `RF_HF_TIMEOUT_S` | Timeout | `30` |
| `RESEARCHFORGE_HF_LIVE_TEST` | Enable live tests | (unset) |

## Test Coverage: 45 passed, 3 skipped (live)
