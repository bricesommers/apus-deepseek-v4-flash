#!/bin/bash
# tools/docker/test-linux.sh — M12a-1 Linux/x86_64 test harness.
#
# Builds the dev image (cached; pip deps are NOT reinstalled unless
# tools/docker/Dockerfile.dev changes), mounts the repo READ-ONLY at /repo,
# copies the build/test inputs (c, tests, tools, reference*, Makefile —
# NOT the 157 GB weights/) into the container's own /src, and runs the
# given make targets there with the system python3.
#
# The read-only mount + in-container copy is deliberate: it keeps ELF build
# artifacts out of the macOS work tree (an in-place Linux build would
# overwrite bin/apus and friends, leaving the Mac side broken).
#
# Usage:
#   tools/docker/test-linux.sh                 # the full portable battery
#   tools/docker/test-linux.sh test-m3 test-m4c
#   tools/docker/test-linux.sh test-m1         # m1 python (unittest) suite
# Env: APUS_LINUX_IMAGE overrides the image tag.
#
# Note: linux/amd64 runs under Rosetta/QEMU emulation on Apple Silicon —
# expect 5-20x slowdowns on compute-heavy suites.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${APUS_LINUX_IMAGE:-apus-linux-dev:m12a1}"

if [ $# -eq 0 ]; then
    set -- test-m2 test-m3 test-m4a test-m4c test-m5 test-m6a test-m6b \
           test-m6c test-m7a test-m8 test-m9a test-m9b test-m9c test-m9d \
           test-m9e test-m11b check-m11a test-m1 test-m12a2
    # m7b (Metal) is macOS-only BY DESIGN — excluded.
fi

echo ">> building image $IMAGE (linux/amd64)"
docker build --platform linux/amd64 -t "$IMAGE" \
    -f "$ROOT/tools/docker/Dockerfile.dev" "$ROOT/tools/docker"

echo ">> running ${*} in-container"
docker run --rm -i --platform linux/amd64 \
    -v "$ROOT:/repo:ro" \
    "$IMAGE" bash -s -- "$@" <<'CONTAINER_EOF'
set -u
mkdir -p /src
cp -a /repo/c /repo/tests /repo/tools /repo/reference /repo/reference-0731 \
      /repo/Makefile /src/
# Purge copied build artifacts: tests/*/bin and bin/ hold macOS (Mach-O)
# binaries fresh enough that make would consider them up-to-date and try to
# EXECUTE them (Exec format error). Deleting them forces clean ELF builds.
find /src -type d -name bin -exec rm -rf {} + 2>/dev/null || true
rm -rf /src/bin
cd /src
pass=""; fail=""
for t in "$@"; do
    echo
    echo "=== $t ==="
    if [ "$t" = "test-m1" ]; then
        # m1 is a python unittest suite, not a make target (tests/m1/README)
        if (cd tests/m1 && python3 -m unittest discover -s . 2>&1 | tail -5 \
            && [ "${PIPESTATUS[0]}" -eq 0 ]); then
            pass="$pass $t"
        else
            fail="$fail $t"
        fi
    elif make -s PY=python3 "$t"; then
        pass="$pass $t"
    else
        fail="$fail $t"
    fi
done
echo
echo "==================== summary ===================="
echo "pass:$pass"
echo "fail:$fail"
[ -z "$fail" ]
CONTAINER_EOF
