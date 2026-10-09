# Marvel's Wolverine on Linux + AMD (RADV)

Work in progress on top of IDXTRI's `wolverine-v1` (KytyPS5 2026-10-03 + the Wolverine work of PR #937),
aimed at Linux with AMD GPUs (Mesa RADV). No game files, keys or firmware are included: you need your own
dump of the game (PPSA03671).

## Status (2026-10-07)

Tested on a Ryzen 7 5800X3D + Radeon RX 7900 XT, Mesa RADV, Linux 7.2, opening jungle area:

- Title, menus and gameplay work; character models render correctly on AMD.
- About 25 fps standing in the jungle; 7-12 fps in heavy combat (fire, many enemies).
- The bottleneck is the single command-processor thread (per-draw resource preparation) and, in heavy
  scenes, GPU serialization (a barrier/render-pass break around most draws). Both are being worked on.

## What this branch adds

AMD/RADV correctness:
- Guest wave32 mesh shaders run one guest wave per pass, pixel shaders get the guest wave size, wave32
  vertex shaders wrap the lane index at 32 (V_MBCNT, Senaxx d7873b2a): no exploded or cracked geometry,
  no rainbow sparks.
- Vertices that export no position are culled instead of undefined; shader clock divisor; mesh draws over
  65535 groups split; BC6H/BC7 3D image flags; LibAtrac9 band-extension bounds (see below).

Performance (command processor), each measured with same-process A/B switches:
- Draw prep (`KYTY_DRAW_PREP`, port of Senaxx 6632a241 onto this branch's resource memo): a scanner
  thread follows each queued submission's registers and evaluates the resource walks of its draws and
  dispatches ahead of the command processor; the command processor takes them as memo hits (same inputs,
  every logged read unchanged). Command processor -11%, +9% fps.
- SRT replay traces (port of Senaxx 598030de/34e492b2) and an in-order replay of the resource walk
  (`KYTY_WALK_INORDER`), texture description cache (Senaxx 50a04054), `FindImage` memo, lazy timeline
  queries, shader-expansion hashing, shared address-space lock for tracker protections, larger fault-ahead
  and readback windows, warm shader cache + asynchronous pipeline compiles (no gameplay stutter from
  shader compiles once warm).
- Almost every change has a `KYTY_*` live switch (environment or `KYTY_LIVE_FILE`) and is marked
  `KYTY_LOCAL_HACK` in the source; research-only switches default to off.

## Build (Ubuntu 26.04 or a distrobox with clang 21 and Qt 6)

```
git submodule update --init --recursive
git -C 3rdparty/LibAtrac9 apply ../../packaging/wolverine/linux-amd/LibAtrac9-band-extension.patch
cmake -S . -B _Build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build _Build/linux --target launcher
cmake --install _Build/linux --prefix _Build/linux/install
```

The LibAtrac9 patch fixes an out-of-bounds write in the audio decoder that crashed the game in combat.
A PGO build (`-fprofile-generate`, a play session, `llvm-profdata merge`, `-fprofile-use`) adds ~15%.

## Run

```
packaging/wolverine/linux-amd/run-wolverine.sh /path/to/PPSA03671-app0
```

The script sets the tuned options (environment switches, `--amd-cpu`, `--bindless`, `--tessellation`,
the skip list of ray-tracing shaders). The first run compiles shaders: expect stutter until the cache is warm.

## Known issues

Some lighting/texture glitches, a full-screen red flash in the Omega Red fight (under investigation),
ray-traced effects skipped, occlusion queries treated as always visible. Save often.

## Credits

KytyPS5 and contributors; IDXTRI (`wolverine-v1`); Mac (itsmemac), Senaxx and Ali Almohaya (PR #937 and the
Wolverine fixes, SRT replay traces); chenxiao07 (draw-record design ideas); Jetsku (ideas from the Astro Bot
work). GPL-2.0, like the rest of the repository.

Not affiliated with Sony, Insomniac Games or Marvel.
