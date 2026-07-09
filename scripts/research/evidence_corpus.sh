#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# EVIDENCE CORPUS — the presentation evidence run.
#
# One structured corpus feeds ~8 claims simultaneously (every run logs its
# full EF trajectory, verdict, fingerprint, moves, grounded citations):
#
#   C1 convergence          <- EF traces of any run that fired
#   C2 within-F learning    <- iterations/starting-E across a species' runs
#   C3 cross-domain transfer<- SEQUENTIAL different-domain runs of one species
#   C4 orthogonality        <- all EF values -> correlation + sole-firing
#   C6 role separability    <- role-labeled node outputs (grounded -> cite revives?)
#   C7 grounded content     <- citations/SOURCES vs archived ungrounded corpus
#   C8 verdict=directive    <- every run (one error model by construction)
#   C9 fingerprint coverage <- species span divergent/convergent/sequential
#
# Design: BLOCKED BY SPECIES, SEQUENTIAL WITHIN — state persists, so run k+1
# of a species inherits run k's learned moves/priors. Since every run in a
# block is a DIFFERENT DOMAIN, the learning curve within a block IS the
# cross-domain transfer measurement (stronger than repeating one task, which
# could be dismissed as memorization). Block boundaries are free negative
# controls: a new species must NOT inherit the previous block's moves.
#
# Usage:
#   ./scripts/evidence_corpus.sh            # archive state, run all 15
#   NO_ARCHIVE=1 ./scripts/evidence_corpus.sh   # keep existing state (resume)
# ---------------------------------------------------------------------------
set -u

LLM_URL=$(python3 -c "from ceo_delta.config import Config; print(Config().llm_base_url)")
# plain curl (no -f) exits 0 on ANY HTTP response, even an error page -- that's
# "the server is alive". It only fails (nonzero) on connection-level problems
# (unreachable host, refused, timeout), which is the actual precondition here.
if ! curl -s -o /dev/null --max-time 5 "$LLM_URL"; then
  echo "ERROR: vLLM backend at $LLM_URL is unreachable. Set CEO_LLM_URL or start the server first."; exit 1
fi

ITERS=$(grep -oP 'max_ceo_eval_iterations:\s*int\s*=\s*\K\d+' ceo_delta/config.py)
if [[ "$ITERS" != "5" ]]; then
  echo "ERROR: max_ceo_eval_iterations is $ITERS, expected 5. Fix config.py first."; exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs/evidence_$TS"
mkdir -p "$LOGDIR"

# Archive prior state: keeps the UNGROUNDED corpus as the C7 contrast baseline,
# and clears the escalation repertoire (incl. the pre-fix poisoned move) so the
# learning curves start clean and uncontaminated.
if [[ -z "${NO_ARCHIVE:-}" && -d .ceo_delta ]]; then
  mv .ceo_delta ".ceo_delta_archive_$TS"
  echo "archived .ceo_delta -> .ceo_delta_archive_$TS (C7 ungrounded baseline)"
fi

typeset -a S2_VERIFY S1_DIVERGE S3_CONVERGE S4_SEQ

# BLOCK S2 — FLAGSHIP: divergent/verification/verification, 5 unrelated domains.
# Deepest block = strongest transfer curve. Domains share zero vocabulary.
S2_VERIFY=(
  "Verify whether eventual consistency is always sufficient for e-commerce inventory systems"
  "Verify whether horizontal pod autoscaling is always sufficient for handling traffic spikes in Kubernetes"
  "Verify whether a plea bargain is always sufficient to resolve a criminal case fairly"
  "Verify whether vaccination alone is always sufficient to prevent disease outbreaks in a population"
  "Verify whether client-side caching is always sufficient for reducing API latency in mobile apps"
)

# BLOCK S1 — divergent/synthesis/artifact, 4 domains (grounding showcase:
# retrieval-heavy, so citations + SOURCES appear here most).
S1_DIVERGE=(
  "What are the key factors that make a distributed database resilient to network partitions"
  "What are the primary factors that determine soil fertility in agricultural land"
  "What are the main drivers of employee retention in large organizations"
  "What are the primary factors influencing urban air quality"
)

# BLOCK S3 — convergent/synthesis: reconcile GIVEN conflicting inputs
# (structurally distinct from gather-then-merge; exercises the merge check).
S3_CONVERGE=(
  "