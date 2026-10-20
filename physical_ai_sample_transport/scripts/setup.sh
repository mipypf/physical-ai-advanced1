#!/usr/bin/env bash
# Ubuntu 24.04 x86_64, Python 3.12, uv, curl and dpkg-deb. No sudo required.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
uv venv --allow-existing .venv --python python3
uv pip install --python .venv/bin/python -r requirements-sim.txt
uv pip install --python .venv/bin/python torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
mkdir -p vendor .local/downloads artifacts
if [[ ! -d vendor/crane_plus/.git ]]; then
  git clone https://github.com/rt-net/crane_plus.git vendor/crane_plus
  git -C vendor/crane_plus checkout 34c4584c7373463f319174bb76a2198943ab9cb8
fi
if [[ "$(git -C vendor/crane_plus rev-parse HEAD)" != 34c4584c7373463f319174bb76a2198943ab9cb8 ]]; then
  echo 'Unexpected CRANE+ revision; please preserve this checkout and use the documented pinned source.' >&2
  exit 1
fi
fetch_deb() {
  local relative="$1" digest="$2" file="$TASK_ROOT/.local/downloads/${1##*/}"
  if [[ ! -f "$file" ]]; then
    curl --fail --location --retry 3 "https://archive.ubuntu.com/ubuntu/pool/main/$relative" -o "$file"
  fi
  echo "$digest  $file" | sha256sum --check -
  dpkg-deb --extract "$file" "$TASK_ROOT/.local"
}
fetch_deb m/mesa/libosmesa6_24.0.5-1ubuntu1_amd64.deb f27463d400dba6d8862060fded4d7ec5be73a837344844cbeb082ef32d269df5
fetch_deb m/mesa/libglapi-mesa_24.0.5-1ubuntu1_amd64.deb 28191867c3a68d59aca31755789328d28960223127b05db6a46b7e8f5646a495
fetch_deb l/llvm-toolchain-17/libllvm17t64_17.0.6-9ubuntu1_amd64.deb 1863b5459ca4a6691f5e039e76157f074ee08b48176267642ee4dfea065f5db4
scripts/run.sh scripts/build_scene.py
printf '%s\n' 'Setup complete. Run scripts/run.sh scripts/vision.py, then scripts/run.sh scripts/demo.py --ai --video'
