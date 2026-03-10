#!/usr/bin/env bash
# Evaluate currently completed dseed=0 tlen_v2 models on CPU.
# Writes a partial phase plot without waiting for all training jobs to finish.

set -eo pipefail

CFD_GPU_PREFIX="/jumbo/yaoqingyang/batman/miniconda3/envs/cfd-gpu"
export LD_LIBRARY_PATH="${CFD_GPU_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/jumbo/yaoqingyang/yuxin/JAX-CFD/jax-cfd:/jumbo/yaoqingyang/yuxin/JAX-CFD/models:/jumbo/yaoqingyang/yuxin/JAX-CFD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"
PYTHON="${CFD_GPU_PREFIX}/bin/python3"
OUT_DIR="${ROOT}/results/tlen_v2_partial_eval_cpu"
LOG_DIR="${ROOT}/logs/eval_tlen_v2_cpu_partial"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

WORKER1_CSV="${OUT_DIR}/phase_metrics_worker1.csv"
WORKER2_CSV="${OUT_DIR}/phase_metrics_worker2.csv"
FINAL_CSV="${OUT_DIR}/phase_metrics.csv"
LIST_FILE="${OUT_DIR}/completed_specs.txt"

rm -f "${WORKER1_CSV}" "${WORKER2_CSV}" "${FINAL_CSV}" "${LIST_FILE}"

echo "=== collect completed dseed=0 models ==="
count=$("${PYTHON}" - <<'PY'
from pathlib import Path

root = Path("/jumbo/yaoqingyang/yuxin/JAX-CFD/models_tlen_v2")
specs = []
for path in sorted(root.glob("re*_dseed0_mseed*")):
    if not path.is_dir():
        continue
    if any("tmp" not in p.name for p in path.glob("checkpoint_*")):
        specs.append(path.name)
specs = sorted(set(specs))
out = Path("/jumbo/yaoqingyang/yuxin/JAX-CFD/results/tlen_v2_partial_eval_cpu/completed_specs.txt")
out.write_text("\n".join(specs) + ("\n" if specs else ""))
print(len(specs))
PY
)
echo "completed models: ${count}"

run_worker() {
  local worker_id=$1
  local worker_csv=$2
  "${PYTHON}" - <<PY | while IFS= read -r spec; do
specs = [line.strip() for line in open("${LIST_FILE}") if line.strip()]
wid = ${worker_id}
for i, spec in enumerate(specs):
    if i % 2 == wid:
        print(spec)
PY
    RE=$(echo "${spec}" | sed -E 's/^re([0-9]+)_tlen([0-9]+)_dseed([0-9]+)_mseed([0-9]+)$/\1/')
    TLEN=$(echo "${spec}" | sed -E 's/^re([0-9]+)_tlen([0-9]+)_dseed([0-9]+)_mseed([0-9]+)$/\2/')
    DSEED=$(echo "${spec}" | sed -E 's/^re([0-9]+)_tlen([0-9]+)_dseed([0-9]+)_mseed([0-9]+)$/\3/')
    MSEED=$(echo "${spec}" | sed -E 's/^re([0-9]+)_tlen([0-9]+)_dseed([0-9]+)_mseed([0-9]+)$/\4/')
    MODEL_DIR="${ROOT}/models_tlen_v2/${spec}"
    TRAIN_NC="${ROOT}/content/kolmogorov_re${RE}/train_2048x2048_64x64.nc"
    EVAL_NC="${ROOT}/content/kolmogorov_re${RE}/long_eval_2048x2048_64x64.nc"
    LOG_FILE="${LOG_DIR}/${spec}.log"
    env JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "${PYTHON}" -u "${ROOT}/scripts/eval_one_model.py" \
      --model_dir "${MODEL_DIR}" \
      --eval_nc "${EVAL_NC}" \
      --train_nc "${TRAIN_NC}" \
      --re "${RE}" --ntraj 32 --tlen "${TLEN}" --seed "${MSEED}" --data_seed "${DSEED}" \
      --output_csv "${worker_csv}" \
      --length 200 --inner_steps 10 \
      > "${LOG_FILE}" 2>&1
    echo "worker${worker_id}: ${spec}"
  done
}

run_worker 0 "${WORKER1_CSV}" &
run_worker 1 "${WORKER2_CSV}" &
wait

echo "=== merge CSVs ==="
"${PYTHON}" - <<'PY'
import csv
from pathlib import Path

paths = [
    Path("/jumbo/yaoqingyang/yuxin/JAX-CFD/results/tlen_v2_partial_eval_cpu/phase_metrics_worker1.csv"),
    Path("/jumbo/yaoqingyang/yuxin/JAX-CFD/results/tlen_v2_partial_eval_cpu/phase_metrics_worker2.csv"),
]
out = Path("/jumbo/yaoqingyang/yuxin/JAX-CFD/results/tlen_v2_partial_eval_cpu/phase_metrics.csv")
rows = []
fieldnames = None
for path in paths:
    if not path.exists():
        continue
    with path.open() as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)

rows.sort(key=lambda r: (int(r["seed"]), int(r["tlen"]), int(r["re"])))
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"merged {len(rows)} rows -> {out}")
PY

echo "=== render phase plot ==="
"${PYTHON}" "${ROOT}/scripts/plot_phase.py" \
  --csv "${FINAL_CSV}" \
  --output_dir "${OUT_DIR}" \
  --yparam tlen \
  --ylabel "Trajectory Length (frames)" \
  --title "Partial phase plot from completed checkpoints (CPU eval)"

echo "done -> ${OUT_DIR}/phase_plot.png"
