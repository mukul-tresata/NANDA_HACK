#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# EF calibration batch runner.
#
# Runs a diagnostic corpus across all fingerprint species + deliberate
# per-axis violators, so scripts/calibrate_ef.py has real data to work with.
# Each run's full output is saved under logs/ and the descent data lands in
# .ceo_delta/runs.jsonl automatically (that's what the diagnostic reads).
#
# Usage:
#   ./run_diagnostic_batch.sh                # baseline x1, violators x2 (default)
#   VIOLATOR_REPEATS=3 ./run_diagnostic_batch.sh
#
# Keep running after closing the terminal:
#   nohup ./run_diagnostic_batch.sh > batch.out 2>&1 &
#   tail -f batch.out          # watch progress
# ---------------------------------------------------------------------------
set -u
setopt shwordsplit 2>/dev/null || true

BASELINE_REPEATS=${BASELINE_REPEATS:-1}
VIOLATOR_REPEATS=${VIOLATOR_REPEATS:-2}
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY not set. export it first, then re-run."
  exit 1
fi

# -- prompt blocks ----------------------------------------------------------
typeset -a BASELINE FLOW PARTITION ROLE SCALE

BASELINE=(
  "What are the key factors that determine the nutritional quality of a human diet"
  "What are the main drivers of employee retention in large organizations"
  "Compare the trade-offs between remote and in-office work models for software teams"
  "Compare monolithic versus microservices architecture for a mid-size SaaS product"
  "Summarize the main causes of the 2008 financial crisis"
  "What are the primary factors influencing urban air quality"
  "Explain the main mechanisms by which vaccines produce immunity"
  "What determines the long-term success of a startup after its first funding round"
)

FLOW=(
  "Walk through the exact ordered steps to synthesize aspirin in a lab, where each step depends on the previous one completing"
  "Describe the precise sequence a bill must follow to become law in the US, where each stage strictly gates the next"
  "Lay out the step-by-step process to brew beer from raw grain to bottling, where each stage must finish before the next begins"
  "Explain the ordered sequence of steps to perform a blood transfusion safely, each step gating the next"
  "A candidate aced the technical interview but failed the culture-fit round — reconcile these two conflicting signals into a single hire/no-hire decision"
  "One financial model says the company is undervalued and another says overvalued — reconcile the two into a single investment recommendation"
  "The patient's MRI suggests a tumor but the biopsy came back benign — reconcile these conflicting results into one clinical course of action"
  "Two A/B tests on the same feature gave opposite results — reconcile them into a single ship/no-ship decision"
)

PARTITION=(
  "What are all the factors that make a city livable"
  "What are all the things that contribute to a person's overall physical health"
  "What are all the aspects that make a video game fun and engaging"
  "What are all the elements that make a movie critically successful"
  "What are all the factors that influence consumer purchasing decisions online"
  "What are all the components of a strong national economy"
)

ROLE=(
  "Design a novel algorithm for detecting fraud in real-time payment streams"
  "Design a new board game that teaches probability to children"
  "Invent a scheduling system for a shared community kitchen used by 50 families"
  "Explain how the quicksort algorithm recursively partitions and sorts an array like [5,2,9,1,7]"
  "Design a compression scheme optimized for storing millions of near-duplicate log files"
  "Create a novel grading rubric for evaluating creativity in student essays"
)

SCALE=(
  "Explain how a recursive descent parser evaluates the nested expression (((1+2)+3)+4)"
  "Explain how merge sort recursively splits and merges the list [8,3,5,1,9,2]"
  "Walk through how a factorial function computes 5! by recursively calling itself"
  "Explain how a file system traverses a deeply nested directory tree to find a file"
  "Describe how a nested set of Russian dolls would be catalogued one layer at a time"
  "In one sentence, define what an API is"
)

# -- runner -----------------------------------------------------------------
COUNT=0
FAILS=0
run_one() {
  local block="$1"; local prompt="$2"
  COUNT=$((COUNT+1))
  local safe=$(echo "$prompt" | tr -c 'a-zA-Z0-9' '_' | cut -c1-40)
  local logfile="$LOGDIR/${COUNT}_${block}_${safe}.log"
  echo "[$(date +%H:%M:%S)] #$COUNT [$block] ${prompt:0:70}..."
  if ! python cli.py run "$prompt" > "$logfile" 2>&1; then
    echo "   ^ FAILED (see $logfile)"
    FAILS=$((FAILS+1))
  fi
}

run_block() {
  local name="$1"; local reps="$2"; shift 2
  local -a prompts=("$@")
  for r in $(seq 1 "$reps"); do
    echo "=== BLOCK $name  (pass $r/$reps) ==="
    for p in "${prompts[@]}"; do run_one "$name" "$p"; done
  done
}

echo "############################################################"
echo "EF CALIBRATION BATCH  |  baseline x$BASELINE_REPEATS, violators x$VIOLATOR_REPEATS"
echo "logs -> $LOGDIR   |   corpus -> .ceo_delta/runs.jsonl"
echo "############################################################"
START=$(date +%s)

run_block "baseline"  "$BASELINE_REPEATS"  "${BASELINE[@]}"
run_block "flow"      "$VIOLATOR_REPEATS"  "${FLOW[@]}"
run_block "partition" "$VIOLATOR_REPEATS"  "${PARTITION[@]}"
run_block "role"      "$VIOLATOR_REPEATS"  "${ROLE[@]}"
run_block "scale"     "$VIOLATOR_REPEATS"  "${SCALE[@]}"

END=$(date +%s)
echo ""
echo "############################################################"
echo "DONE: $COUNT runs, $FAILS failures, $(( (END-START)/60 )) min elapsed"
echo "############################################################"
echo ""
echo "=== CALIBRATION DIAGNOSTIC ==="
python scripts/calibrate_ef.py
