#!/usr/bin/env bash
# Build the installers for macOS and Linux.
#   bash installer/build_unix_installers.sh
# Output (installer/Output), <v> = the version in pro_ui/help_text.py:
#   BehavioralTracker_Setup_<v>_linux.run            BehavioralTracker_Setup_<v>_linux_licensed.run          (online)
#   BehavioralTracker_Setup_<v>_linux_offline.run    BehavioralTracker_Setup_<v>_linux_licensed_offline.run  (offline)
#   BehavioralTracker_Setup_<v>_mac.zip              BehavioralTracker_Setup_<v>_mac_licensed.zip            (online)
# online  = small file, downloads Python + packages while installing
# offline = large file, Python + every package already inside (no internet needed; built once and cached in
#           installer/build/linux_offline -- made again when requirements-unix.txt, unix/slim.py or
#           unix/trim_linux.py change; needs binutils for strip/readelf)
# The app inside is the same encrypted bundle as the Windows Setup (protect/build_secure.py, Python 3.12 bytecode).
# Options (environment):  EDITIONS="licensed"  (default: "unlicensed licensed")
#                         ONLINE_ONLY=1        (only the online files: no offline runtime, nothing downloaded here
#                                               except what the build itself needs) -- used by the GitHub build
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
WORK="$HERE/build/unix"; OUT="$HERE/Output"
PYSECURE="${PYSECURE:-python3.12}"
VER="$(grep -m1 '^VERSION' "$ROOT/pro_ui/help_text.py" | cut -d'"' -f2)"
EDITIONS="${EDITIONS:-unlicensed licensed}"
ONLINE_ONLY="${ONLINE_ONLY:-0}"
rm -rf "$WORK"; mkdir -p "$WORK" "$OUT"

# ---------------------------------------------------------------- offline Linux runtime (cached)
# Python 3.12 + every package, X11 libraries added and already slimmed, packed as one runtime.tar.xz.
# Wheels are limited to glibc 2.34 (Ubuntu 22.04, Debian 12, RHEL 9, Fedora 35 or newer) -- the installer checks it.
LPY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.12.7+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz"
LPY_SHA="43576f7db1033dd57b900307f09c2e86f371152ac8a2607133afa51cbfc36064"
CACHE="$HERE/build/linux_offline"; RT="$CACHE/runtime.tar.xz"
RTHASH="$(cat "$HERE/requirements-unix.txt" "$HERE/unix/slim.py" "$HERE/unix/trim_linux.py" "$HERE"/unix/xlibs/* | sha256sum | cut -c1-16)"
if [ "$ONLINE_ONLY" != 1 ] && { [ ! -f "$RT" ] || [ ! -f "$CACHE/.rt_$RTHASH" ]; }; then
  echo "== Offline Linux runtime (Python 3.12 + packages)"
  mkdir -p "$CACHE"; rm -rf "${CACHE:?}/rt" "${CACHE:?}/wheels" "$CACHE"/.rt_* "$RT"; mkdir -p "$CACHE/rt"
  [ -f "$CACHE/python-linux.tar.gz" ] || curl -fsSL --retry 3 -o "$CACHE/python-linux.tar.gz" "$LPY_URL"
  echo "$LPY_SHA  $CACHE/python-linux.tar.gz" | sha256sum -c --quiet -
  tar xzf "$CACHE/python-linux.tar.gz" -C "$CACHE/rt"
  RPY="$CACHE/rt/python/bin/python3"
  PLAT=(); for v in 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17; do PLAT+=(--platform "manylinux_2_${v}_x86_64"); done
  "$RPY" -E -s -m pip download --disable-pip-version-check -q --only-binary=:all: "${PLAT[@]}" \
      --platform manylinux2014_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64 \
      --python-version 3.12 --implementation cp -d "$CACHE/wheels" -r "$HERE/requirements-unix.txt"
  "$RPY" -E -s -m pip install --disable-pip-version-check --no-warn-script-location --no-cache-dir -q \
      --no-index --find-links "$CACHE/wheels" -r "$HERE/requirements-unix.txt"
  SITE="$("$RPY" -E -s -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cp "$HERE"/unix/xlibs/* "$SITE/PySide6/Qt/lib/"
  rm -rf "$CACHE/rt/python/share" "$CACHE"/rt/python/lib/python3.12/test "$CACHE"/rt/python/lib/python3.12/idlelib
  "$RPY" -E -s "$HERE/unix/slim.py" "$SITE"
  python3 "$HERE/unix/trim_linux.py" "$CACHE/rt/python"        # debug info, unused Qt modules / plugins / libraries
  find "$CACHE/rt/python" -name '__pycache__' -prune -exec rm -rf {} +
  tar -C "$CACHE/rt" -cf - python | xz -T0 -6 > "$RT.tmp" && mv "$RT.tmp" "$RT"
  rm -rf "${CACHE:?}/rt" "${CACHE:?}/wheels"; touch "$CACHE/.rt_$RTHASH"
fi
# app icon for macOS (.icns) + Linux (.png)
python3 - "$ROOT/pro_ui/icons/app_logo.png" "$WORK/app.icns" <<'PY'
import sys
from PIL import Image
Image.open(sys.argv[1]).convert("RGBA").save(sys.argv[2])
PY
for kind in $EDITIONS; do
  flags=(); [ "$kind" = licensed ] && flags=(--licensed)
  "$PYSECURE" "$ROOT/protect/build_secure.py" --src "$ROOT" --out "$WORK/app_$kind" "${flags[@]}" > /dev/null
  find "$WORK/app_$kind" -name '__pycache__' -prune -exec rm -rf {} +
  suffix=""; [ "$kind" = licensed ] && suffix="_licensed"
  for target in linux mac; do
    P="$WORK/payload_${target}_$kind"; mkdir -p "$P"
    cp -R "$WORK/app_$kind" "$P/app"
    cp "$HERE/requirements-unix.txt" "$HERE/unix/slim.py" "$P/"
    if [ "$target" = linux ]; then
      cp -R "$HERE/unix/xlibs" "$P/xlibs"
      cp "$ROOT/pro_ui/icons/app_logo.png" "$P/behavioraltracker.png"
    else
      cp "$HERE/requirements-mac-intel.txt" "$P/"
      cp "$WORK/app.icns" "$P/app.icns"
    fi
    S="$WORK/setup_${target}_$kind"
    sed -e "s/@EDITION@/$kind/" -e "s/@TARGET@/$target/" -e "s/@VERSION@/$VER/" "$HERE/unix/install_bt.sh" > "$S.head"
    cp "$S.head" "$S"
    tar -C "$P" -czf - . >> "$S"
    chmod +x "$S"
    if [ "$target" = linux ]; then
      cp "$S" "$OUT/BehavioralTracker_Setup_${VER}_linux$suffix.run"
      [ "$ONLINE_ONLY" = 1 ] && continue
      cp "$RT" "$P/runtime.tar.xz"                       # offline: the ready runtime travels inside too
      cp "$S.head" "$S.offline"
      tar -C "$P" -czf - . >> "$S.offline"
      chmod +x "$S.offline"
      cp "$S.offline" "$OUT/BehavioralTracker_Setup_${VER}_linux${suffix}_offline.run"
    else
      M="$WORK/mac_$kind/BehavioralTracker Setup"; mkdir -p "$M"
      cp "$S" "$M/Install BehavioralTracker.command"
      cat > "$M/How to install.txt" <<TXT
BehavioralTracker $VER for macOS (macOS 13 Ventura or newer, Apple Silicon or Intel)

1. Double-click "Install BehavioralTracker.command".
   The first time macOS blocks it ("cannot be opened" / "could not verify"):
   click Done, open System Settings > Privacy & Security, scroll down,
   click "Open Anyway" and enter your Mac password.
2. A Terminal window opens and installs BehavioralTracker. It downloads the
   program components (about 250 MB) - keep the Mac connected to the internet.
   This takes about 5-10 minutes.
3. Open BehavioralTracker from Launchpad, Spotlight, or Applications in your home folder.

To remove it: drag ~/Applications/BehavioralTracker.app to the Trash.
Settings, history and licence are in ~/.behavioraltracker
Support: manojkumarmunusamy1101@outlook.com
TXT
      (cd "$WORK/mac_$kind" && rm -f "$OUT/BehavioralTracker_Setup_${VER}_mac$suffix.zip" && \
       zip -qry "$OUT/BehavioralTracker_Setup_${VER}_mac$suffix.zip" "BehavioralTracker Setup")
    fi
  done
done
ls -la "$OUT" | grep -E "linux|mac"
