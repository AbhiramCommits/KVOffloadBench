#!/usr/bin/env bash
# Real loopback NVMe-oF/TCP target + host, for the nvmeof_block tier.
#
# Creates an nvmet subsystem backed by a sparse file, exports it on 127.0.0.1:4420,
# connects the local host with nvme-tcp, and prints the resulting block device.
# Requires Linux with nvmet, nvmet-tcp and nvme-tcp modules, and root.
#
#   sudo scripts/setup_nvmeof_tcp.sh
#   export KVOB_NVMEOF_DEV=/dev/nvme1n1    # printed by the script
#
# RDMA/RoCE variant (BlueField-3 or any RDMA NIC):
#   1. Install rdma-core and load the vendor driver (mlx5 for ConnectX/BlueField).
#   2. Replace the TCP port attributes below with:
#        echo rdma > /sys/kernel/config/nvmet/ports/1/addr_trtype
#        echo <server-ip> > /sys/kernel/config/nvmet/ports/1/addr_traddr
#        echo ipv4 > .../addr_adrfam
#        echo 4420 > .../addr_trsvcid
#   3. Connect from the host:
#        nvme connect -t rdma -n "$NQN" -a <server-ip> -s 4420
#   4. Verify with `nvme list-subsys` and `nvme list`.
#   No firewall rule is needed on loopback; for a remote server open 4420/tcp
#   (default RoCE traffic uses the NIC's rdma_cm service id, not TCP).
#
# The tier is destructive: it owns the first KVOB_NVMEOF_SIZE_GIB of the
# namespace. Point it only at this scratch export.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: this script must run as root (sudo)" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: NVMe-oF setup requires Linux; this host is $(uname -s)" >&2
  exit 1
fi

BACKING="${KVOB_NVMEOF_BACKING:-/var/lib/kvoffloadbench/nvmeof-backing.img}"
SIZE_GIB="${KVOB_NVMEOF_SIZE_GIB:-8}"
NQN="nqn.2026-01.io.kvoffloadbench:storage"
HOST_NQN="nqn.2026-01.io.kvoffloadbench:host"
PORT="${KVOB_NVMEOF_PORT:-4420}"
TRANSPORT="${KVOB_NVMEOF_TRANSPORT:-tcp}"
TRADDR="${KVOB_NVMEOF_TRADDR:-127.0.0.1}"

CONFIGFS=/sys/kernel/config/nvmet
mkdir -p "${CONFIGFS}"
if ! mountpoint -q /sys/kernel/config; then
  mount -t configfs none /sys/kernel/config
fi

echo "==> loading kernel modules"
modprobe nvmet
if [[ "${TRANSPORT}" == "tcp" ]]; then
  modprobe nvmet-tcp
  modprobe nvme-tcp
else
  modprobe nvmet-rdma
  modprobe nvme-rdma
fi

echo "==> creating backing file ${BACKING} (${SIZE_GIB} GiB sparse)"
mkdir -p "$(dirname "${BACKING}")"
if [[ ! -f "${BACKING}" ]]; then
  truncate -s "${SIZE_GIB}G" "${BACKING}"
fi

echo "==> creating subsystem ${NQN}"
mkdir -p "${CONFIGFS}/subsystems/${NQN}"
echo 1 > "${CONFIGFS}/subsystems/${NQN}/attr_allow_any_host"
mkdir -p "${CONFIGFS}/subsystems/${NQN}/namespaces/1"
echo -n "${BACKING}" > "${CONFIGFS}/subsystems/${NQN}/namespaces/1/device_path"
echo 1 > "${CONFIGFS}/subsystems/${NQN}/namespaces/1/enable"

echo "==> creating ${TRANSPORT} port ${PORT} on ${TRADDR}"
mkdir -p "${CONFIGFS}/ports/1"
echo "${TRANSPORT}" > "${CONFIGFS}/ports/1/addr_trtype"
echo "${TRADDR}" > "${CONFIGFS}/ports/1/addr_traddr"
echo ipv4 > "${CONFIGFS}/ports/1/addr_adrfam"
echo "${PORT}" > "${CONFIGFS}/ports/1/addr_trsvcid"
ln -sfn "${CONFIGFS}/subsystems/${NQN}" "${CONFIGFS}/ports/1/subsystems/${NQN}"

echo "==> connecting the local host"
# Idempotent: disconnect an existing controller for this subsystem first.
if nvme list-subsys 2>/dev/null | grep -q "${NQN}"; then
  echo "    controller already connected; disconnecting to re-connect cleanly"
  nvme disconnect -n "${NQN}" >/dev/null 2>&1 || true
  sleep 1
fi
if [[ "${TRANSPORT}" == "tcp" ]]; then
  nvme connect -t tcp -n "${NQN}" -a "${TRADDR}" -s "${PORT}"
else
  nvme connect -t rdma -n "${NQN}" -a "${TRADDR}" -s "${PORT}"
fi
sleep 1

DEV="$(ls -1 /dev/nvme*n* 2>/dev/null | grep -v 'nvme[0-9]*c' | tail -n 1 || true)"
if [[ -z "${DEV}" ]]; then
  echo "error: nvme connect succeeded but no /dev/nvmeXnY namespace appeared" >&2
  nvme list-subsys || true
  exit 1
fi

echo
echo "NVMe-oF namespace ready: ${DEV}"
echo "Export it for the benchmark:"
echo "  export KVOB_NVMEOF_DEV=${DEV}"
echo "  kvoffloadbench bench-tiers --config configs/tiers_micro.yaml"
echo
nvme list | sed -n '1,5p'
