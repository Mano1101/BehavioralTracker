#!/usr/bin/env bash
# Build the Windows Setup files of BehavioralTracker 1.0.0 on Linux (Ubuntu / WSL) -- no Windows needed.
#
#   sudo apt-get install -y nsis mingw-w64 python3.12 python3-pip     (once)
#   bash installer/build_installer.sh [online|offline|both]
#
# Which editions are built is controlled by KINDS (default: the two protected editions):
#   KINDS="unlicensed licensed"   bash installer/build_installer.sh both
#   KINDS="plain"                 bash installer/build_installer.sh both   # unprotected (source visible)
#
#   plain       : no protection (plain .py source ships)         -> ..._plain*.exe (or no suffix)
#   unlicensed  : protected, runs on any computer                -> ..._unlicensed*.exe
#   licensed    : protected + node-locked licence required       -> ..._licensed*.exe
#
# online  = small Setup, downloads the Python packages while installing
# offline = large Setup, everything inside (no internet needed)
#
# Code signing (Authenticode, optional -- needs a code-signing certificate; sudo apt-get install osslsigncode):
#   SIGN_PFX=/path/certificate.pfx SIGN_PASS=... [SIGN_TS=http://timestamp.digicert.com | none] bash ...
# signs the program launcher, the uninstaller and every Setup file (Windows SmartScreen / antivirus trust them).
set -euo pipefail
MODE="${1:-both}"
KINDS="${KINDS:-unlicensed licensed}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$HERE/build}"
OUTDIR="$HERE/Output"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.12.7+20241016-x86_64-pc-windows-msvc-install_only.tar.gz"
REQ="$HERE/requirements-win.txt"
PYSECURE="${PYSECURE:-python3.12}"        # must be CPython 3.12 (matches the bundled runtime's bytecode)
VER="$(grep -m1 '^VERSION' "$ROOT/pro_ui/help_text.py" | cut -d'"' -f2)"
WHEEL_ARGS=(--platform win_amd64 --python-version 3.12 --implementation cp --abi cp312 --only-binary=:all:)
mkdir -p "$WORK" "$OUTDIR"

# --------------------------------------------------------------------------- base Python bundle (cached)
BASEPY="$WORK/basepy"
if [ ! -f "$BASEPY/python/python.exe" ]; then
  echo "== Base Windows Python 3.12 + pip + launcher"
  rm -rf "${BASEPY:?}"; mkdir -p "$BASEPY"
  [ -f "$WORK/python-win.tar.gz" ] || curl -sSL -o "$WORK/python-win.tar.gz" "$PY_URL"
  tar xzf "$WORK/python-win.tar.gz" -C "$BASEPY"
  find "${BASEPY:?}/python" -name "*.pdb" -delete
  rm -rf "${BASEPY:?}/python/Lib/test" "${BASEPY:?}/python/Lib/idlelib" "${BASEPY:?}/python/Lib/lib2to3"
  rm -rf "${BASEPY:?}"/python/Lib/site-packages/pip "${BASEPY:?}"/python/Lib/site-packages/pip-*.dist-info
  python3 -m pip install --no-cache-dir --target "$BASEPY/python/Lib/site-packages" "${WHEEL_ARGS[@]}" pip
  find "${BASEPY:?}/python" -type d \( -name test -o -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
  python3 "$HERE/trim_bundle.py" "$BASEPY/python" --keep-pip
fi

# launcher + artwork (cached)
if [ ! -f "$BASEPY/BehavioralTracker.exe" ] || [ "$HERE/launcher.c" -nt "$BASEPY/BehavioralTracker.exe" ] \
   || [ "$HERE/launcher.rc" -nt "$BASEPY/BehavioralTracker.exe" ]; then
  python3 "$HERE/make_assets.py" "$WORK"
  cp "$HERE/launcher.rc" "$WORK/launcher.rc"
  (cd "$WORK" && x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher.res)
  x86_64-w64-mingw32-gcc -O2 -s -municode -mwindows -o "$BASEPY/BehavioralTracker.exe" "$HERE/launcher.c" "$WORK/launcher.res" -lshell32
  cp "$WORK/app.ico" "$BASEPY/app.ico"
fi

# offline package set (cached) -- a fully-populated site-packages we can copy per edition
OFFPKG="$WORK/offpkg"
REQHASH="$(sha256sum "$REQ" | cut -c1-16)"          # the cache is made again when the package list changes
if { [ "$MODE" = "offline" ] || [ "$MODE" = "both" ]; } && \
   { [ ! -d "$OFFPKG/cryptography" ] || [ ! -f "$OFFPKG/.req_$REQHASH" ]; }; then   # online Setup: not needed
  echo "== Offline package set (PySide6, OpenCV, NumPy, pandas, SciPy, cryptography, ...)"
  rm -rf "${OFFPKG:?}"; mkdir -p "$OFFPKG"
  python3 -m pip install --no-cache-dir --target "$OFFPKG" "${WHEEL_ARGS[@]}" -r "$REQ"
  touch "$OFFPKG/.req_$REQHASH"
fi

# --------------------------------------------------------------------------- per-edition app
build_app() {       # $1 = kind (plain|unlicensed|licensed) -> echoes the app dir
  local kind="$1"
  local appdir="$WORK/app_$kind"
  rm -rf "$appdir"
  if [ "$kind" = "plain" ]; then
    mkdir -p "$appdir"
    cp -r "$ROOT/main_pro.py" "$ROOT/pro_ui" "$ROOT/tracking" "$ROOT/analysis" "$ROOT/output" \
          "$ROOT/resources" "$ROOT/README_UX.md" "$appdir/"
    find "$appdir" -name __pycache__ -prune -exec rm -rf {} +
  else
    local flags=(); [ "$kind" = "licensed" ] && flags=(--licensed)
    "$PYSECURE" "$ROOT/protect/build_secure.py" --src "$ROOT" --out "$appdir" "${flags[@]}" >&2
  fi
  cp "$REQ" "$appdir/requirements-win.txt"
  echo "$appdir"
}

suffix() { case "$1" in plain) echo "";; *) echo "_$1";; esac; }

SIGN_PFX="${SIGN_PFX:-}"
SIGN_TS="${SIGN_TS:-http://timestamp.digicert.com}"
sign_exe() {        # $1 = .exe signed in place (nothing happens without SIGN_PFX)
  [ -n "$SIGN_PFX" ] || return 0
  local f="$1" ts=()
  [ "$SIGN_TS" != "none" ] && ts=(-t "$SIGN_TS")
  osslsigncode sign -pkcs12 "$SIGN_PFX" ${SIGN_PASS:+-pass "$SIGN_PASS"} -h sha256 \
      -n "BehavioralTracker" -i "https://github.com/Mano1101/BehavioralTracker" "${ts[@]}" \
      -in "$f" -out "$f.signed" >&2
  mv -f "$f.signed" "$f"
}
NSIS_SIGN=()
if [ -n "$SIGN_PFX" ]; then                       # the uninstaller is written by the Setup: sign it while building
  TSOPT=""; [ "$SIGN_TS" != "none" ] && TSOPT="-t $SIGN_TS"
  NSIS_SIGN=(-DSIGNCMD="osslsigncode sign -pkcs12 \"$SIGN_PFX\" ${SIGN_PASS:+-pass \"$SIGN_PASS\"} -h sha256 -n BehavioralTracker $TSOPT -in \"%1\" -out \"%1.s\" && mv -f \"%1.s\" \"%1\"")
fi

stage_for() {       # $1 = kind, $2 = online|offline -> echoes stage dir (BehavioralTracker)
  local kind="$1"
  local smode="$2"
  local st="$WORK/stage_${kind}_${smode}/BehavioralTracker"
  rm -rf "$WORK/stage_${kind}_${smode}"; mkdir -p "$st/Sample video"
  cp -r "$BASEPY/python" "$st/python"
  cp "$BASEPY/BehavioralTracker.exe" "$BASEPY/app.ico" "$st/"
  sign_exe "$st/BehavioralTracker.exe"
  cp -r "$(build_app "$kind")" "$st/app"
  cp "$ROOT/dummy_behavior_test.mp4" "$st/Sample video/" 2>/dev/null || true
  if [ "$smode" = "offline" ]; then
    cp -r "$OFFPKG"/* "$st/python/Lib/site-packages/"
    find "${st:?}/python" -type d \( -name test -o -name tests -o -name __pycache__ \) -prune -exec rm -rf {} +
    python3 "$HERE/trim_bundle.py" "$st/python" >&2
  fi
  echo "$st"
}

for kind in $KINDS; do
  SUF="$(suffix "$kind")"
  if [ "$MODE" = "online" ] || [ "$MODE" = "both" ]; then
    echo "== Online Setup [$kind]"
    ST="$(stage_for "$kind" online)"
    EST_KB=$(( $(du -sk "$ST" | cut -f1) + 760000 ))
    makensis -V2 -DONLINE -DSTAGE="$ST" -DASSETS="$WORK" -DESTSIZE="$EST_KB" "${NSIS_SIGN[@]}" \
        -DOUTFILE="$OUTDIR/BehavioralTracker_Setup_${VER}${SUF}.exe" "$HERE/BehavioralTracker.nsi"
    sign_exe "$OUTDIR/BehavioralTracker_Setup_${VER}${SUF}.exe"
  fi
  if [ "$MODE" = "offline" ] || [ "$MODE" = "both" ]; then
    echo "== Offline Setup [$kind]"
    ST="$(stage_for "$kind" offline)"
    EST_KB=$(( $(du -sk "$ST" | cut -f1) * 115 / 100 ))
    makensis -V2 -DSTAGE="$ST" -DASSETS="$WORK" -DESTSIZE="$EST_KB" "${NSIS_SIGN[@]}" \
        -DOUTFILE="$OUTDIR/BehavioralTracker_Setup_${VER}${SUF}_offline.exe" "$HERE/BehavioralTracker.nsi"
    sign_exe "$OUTDIR/BehavioralTracker_Setup_${VER}${SUF}_offline.exe"
  fi
done
ls -la "$OUTDIR"
