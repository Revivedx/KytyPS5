#!/usr/bin/env bash
# Runs Marvel's Wolverine (PPSA03671) with the settings tuned on Linux + AMD (RADV).
# Usage: run-wolverine.sh <game folder with eboot.bin> [install dir, default _Build/linux/install]
# Optional env: FULLSCREEN=0 (window), KYTY_WARM_CACHE=<file> (shader warm cache), EXTRAENV="A=1 B=2".
game=${1:?usage: run-wolverine.sh <game folder> [install dir]}
dir=${2:-$(dirname "$0")/../../../_Build/linux/install}
fs=(); [ "${FULLSCREEN:-1}" = 1 ] && fs=(--fullscreen)
cd "$dir" || exit 1
exec env KYTY_ASYNC_SHADERS=1 KYTY_NO_RUMBLE=1 KYTY_OVERLAP_CHECK=0 KYTY_ASYNC_WRITE_READBACK=1 \
  KYTY_FAULT_AHEAD=256 KYTY_READBACK_WINDOW_KB=4096 \
  KYTY_GPU_DATA_READS=e5c3f328a12958b1,c6f8be7be2fe2b58 KYTY_MESH_INDIRECT_GPU=1 \
  KYTY_MAPPED_DEVICE_BUFFERS=1 KYTY_DIRECT_UPLOAD=1 KYTY_HOT_PAGES=0 KYTY_RESOURCE_MEMO=1 \
  KYTY_IR_VALIDATE=0 ${EXTRAENV} ./kyty_emulator --game "$game" \
  --bindless --tessellation --readback-linear-images true --amd-cpu "${fs[@]}" \
  --skip-shaders bad108e74fb72e9f,4e7f2c6bb9b158a1,8bfd230b9cd875a2,c4df2a00067e0666,dc76e1223a9bf673,e6d76d24f59f8015
