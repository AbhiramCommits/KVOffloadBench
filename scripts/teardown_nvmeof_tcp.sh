#!/usr/bin/env bash
# Tear down the loopback NVMe-oF target created by setup_nvmeof_tcp.sh.
# Idempotent: safe to run when nothing is configured.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: this script must run as root (sudo)" >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: NVMe-oF teardown requires Linux; this host is $(uname -s)" >&2
  exit 1
fi

NQN="nqn.2026-01.io.kvoffloadbench:storage"
CONFIGFS=/sys/kernel/config/nvmet

echo "==> disconnecting ${NQN}"
nvme disconnect -n "${NQN}" >/dev/null 2>&1 || true

if [[ -d "${CONFIGFS}" ]]; then
  echo "==> removing configfs objects"
  rm -f "${CONFIGFS}/ports/1/subsystems/${NQN}" 2>/dev/null || true
  rmdir "${CONFIGFS}/ports/1" 2>/dev/null || true
  echo 0 > "${CONFIGFS}/subsystems/${NQN}/namespaces/1/enable" 2>/dev/null || true
  rmdir "${CONFIGFS}/subsystems/${NQN}/namespaces/1" 2>/dev/null || true
  rmdir "${CONFIGFS}/subsystems/${NQN}" 2>/dev/null || true
fi

echo "==> done (backing file is left in place; delete it manually if desired)"
