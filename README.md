# Lotto 6/45 V4.0 — Governed Research Platform

V4.0 consolidates the V3.8 engine into a governed research workflow: immutable selection snapshots, append-only experiment events, strict manual draw append, paired draw-level forward evaluation, and explicit model-promotion gates. The existing V3.6.1, V3.7 and V3.8 model research tools remain available for compatibility.

## Data policy

- Draw history is manual CSV / XLSX / XLS import only.
- No Korean Lotto official website, official API, scraping, third-party result mirror, scheduled result refresh, or hidden draw-data fetch is permitted.
- GitHub API support in v40_store.py is solely for storing the user's own experiment metadata and forward-lock artifacts in a separate research-data branch. It is not used to retrieve lottery results.
- Never overwrite data/lotto645.csv, data/sync_status.json, or data/experiment_ledger.json during an application-code deployment.

## V4.0 foundations

1. Canonical data identity — normalized CSV SHA-256 fingerprints for complete history and for the fixed selection window.
2. Strict append — append_draw accepts only the next draw, validates six distinct main numbers and a distinct bonus, rejects conflicting duplicates and gaps, and treats an identical duplicate as idempotent.
3. Selection snapshot identity — research results are checked against the fixed V3.8 retrospective selection slice (through draw 1088), not the ever-growing full history. New results can be scored without silently re-optimizing the frozen model.
4. Append-only event archive — model snapshots, manual data additions and pre-draw locks are versioned JSON events. Archives can be exported and restored; lock import is allow-listed and validates ticket records.
5. Forward evidence — paired comparisons aggregate at draw level, then use moving-block bootstrap confidence intervals and Holm correction. Ticket rows are not treated as independent draws.
6. Promotion governance — the default promotion gate requires at least 100 forward draws, all three primary point estimates positive, all block-bootstrap 95% confidence intervals above zero, and all Holm-adjusted p-values below 0.05. This is a conservative research governance rule, not a claim that an independent fair lottery can be predicted.

## Persistent research storage (optional)

The app works without a token and provides export/import JSON archives. To persist research events in a separate GitHub branch across Streamlit restarts, add a Streamlit secret:

    GITHUB_RESEARCH_TOKEN = YOUR_FINE_GRAINED_TOKEN

The token should be scoped to this repository with Contents read/write permission. The application only writes v40_store/events.jsonl on research-data; it does not modify main or the draw CSV. Keep the token out of the repository and never paste it into chat.

## Tests

From the complete repository root:

    python -m compileall -q .
    python -m unittest discover -s tests -v

The V4.0 tests cover strict append, canonical fingerprints, block-bootstrap reproducibility, multiple-comparison correction, event archive round-tripping, forward-lock recovery and evidence gates. Synthetic tests validate software behavior only; they do not establish predictive advantage.

## Deployment

The V4.0 work is prepared on branch v4.0-architecture first. Review and merge its pull request only after GitHub Actions passes. The production main branch remains the V3.8 rollback baseline until then.