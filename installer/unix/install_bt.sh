#!/bin/bash
# BehavioralTracker @VERSION@ Setup for macOS and Linux.
# Online Setup: downloads Python and the program components during installation.
# Offline Setup (Linux): carries them inside (runtime.tar.xz) -- no internet needed. Then installs the app:
#   macOS : ~/Applications/BehavioralTracker.app      Linux : ~/.local/share/BehavioralTracker + menu entry
# No administrator password needed.  (bash 3.2 compatible -- the bash that ships with macOS)
APP="BehavioralTracker"; VER="@VERSION@"; EDITION="@EDITION@"; TARGET="@TARGET@"
PYREL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016"
PYVER="3.12.7+20241016"
SUPPORT="manojkumarmunusamy1101@outlook.com"
OS="$(uname -s)"; ARCH="$(uname -m)"
DATA="$HOME/.behavioraltracker"; mkdir -p "$DATA" 2>/dev/null
LOG="$DATA/install.log"; : > "$LOG"
SELF="$0"

say()  { printf '%s\n' "$*"; }
step() { printf '\n  > %s\n' "$*"; echo "== $*" >> "$LOG"; }
finish_pause() {
  if [ "$OS" = "Darwin" ] || [ -t 0 ]; then printf '\n  Press Enter to close this window. '; read -r _ < /dev/tty 2>/dev/null; fi
}
fail() {
  printf '\n  X  %s\n' "$*"
  printf '     If it keeps failing, email %s and attach:\n     %s\n' "$SUPPORT" "$LOG"
  finish_pause; exit 1
}
spin() {   # spin <pid> : moving dots while a long step runs
  n=0
  while kill -0 "$1" 2>/dev/null; do
    n=$((n + 1)); printf '.'; [ $((n % 50)) -eq 0 ] && printf '\n    '; sleep 2
  done
  wait "$1"
}
download() {   # download <url> <file>
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 --connect-timeout 30 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
  else python3 -c 'import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
  fi
}

clear 2>/dev/null
say ""
say "  ============================================================"
say "     $APP $VER  -  Setup"
say "     Animal behaviour tracking and analysis"
say "  ============================================================"

# ---------------------------------------------------------------- which computer?
case "$OS/$ARCH" in
  Linux/x86_64)  PYT="x86_64-unknown-linux-gnu"; REQ="requirements-unix.txt" ;;
  Darwin/arm64)  PYT="aarch64-apple-darwin";     REQ="requirements-unix.txt" ;;
  Darwin/x86_64) PYT="x86_64-apple-darwin";      REQ="requirements-mac-intel.txt" ;;
  *) fail "This computer ($OS $ARCH) is not supported. Use the Windows Setup, or a Mac / 64-bit Linux computer." ;;
esac
if [ "$TARGET" = "mac" ] && [ "$OS" != "Darwin" ]; then fail "This is the Mac installer. Please use the Linux installer on this computer."; fi
if [ "$TARGET" = "linux" ] && [ "$OS" != "Linux" ]; then fail "This is the Linux installer. Please use the Mac installer on this computer."; fi
if [ "$OS" = "Darwin" ]; then
  MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
  [ "${MAJOR:-0}" -ge 13 ] 2>/dev/null || fail "$APP needs macOS 13 (Ventura) or newer. This Mac has macOS $(sw_vers -productVersion)."
  DEST="$HOME/Applications/$APP.app"; ROOT="$DEST/Contents/Resources"
else
  DEST="$HOME/.local/share/$APP"; ROOT="$DEST"
  GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"      # e.g. 2.35 (empty if not glibc)
  if [ -n "$GLIBC" ]; then
    GMAJ="${GLIBC%%.*}"; GMIN="${GLIBC#*.}"; GMIN="${GMIN%%.*}"
    if [ "$GMAJ" -lt 2 ] 2>/dev/null || { [ "$GMAJ" -eq 2 ] && [ "$GMIN" -lt 34 ]; } 2>/dev/null; then
      fail "$APP needs a newer Linux (Ubuntu 22.04, Debian 12, Fedora 35, RHEL 9 or newer). This one has glibc $GLIBC."
    fi
  fi
fi

# ---------------------------------------------------------------- unpack the app carried inside this file
step "Preparing the installation..."
WORK="$(mktemp -d 2>/dev/null || mktemp -d -t btsetup)"
trap 'rm -rf "$WORK"' EXIT
LINE="$(awk '/^__BT_PAYLOAD_BELOW__$/ { print NR + 1; exit }' "$SELF")"
[ -n "$LINE" ] || fail "This installer file is damaged. Please download it again."
tail -n +"$LINE" "$SELF" | tar xzf - -C "$WORK" >> "$LOG" 2>&1 || fail "This installer file is damaged. Please download it again."
OFFLINE=0; [ -f "$WORK/runtime.tar.xz" ] && OFFLINE=1
say ""
say "  It will be installed in:  $DEST"
if [ "$OFFLINE" = 1 ]; then
  say "  Everything needed is inside this installer - no internet connection is needed."
else
  say "  Some program components are downloaded now - keep this computer connected to the internet."
fi

if [ "$OFFLINE" = 1 ]; then
# ---------------------------------------------------------------- python + packages, from inside this file
step "Installing program components (about a minute)..."
printf '    '
rm -rf "$DEST"
mkdir -p "$ROOT" || fail "Cannot write to $ROOT"
( xz -dc "$WORK/runtime.tar.xz" | tar xf - -C "$ROOT" >> "$LOG" 2>&1 ) & spin $! \
  || fail "The program components could not be unpacked (is 'xz' installed? It is part of every standard Linux). Details: $LOG"
PY="$ROOT/python/bin/python3"
[ -x "$PY" ] || fail "The program components are incomplete. Please download the installer again."
"$PY" -E -s -c 'import PySide6.QtCore, cv2, numpy, pandas, scipy, cryptography' >> "$LOG" 2>&1 \
  || fail "The program components do not start on this computer. Details: $LOG"
else

# ---------------------------------------------------------------- python (part 1)
step "Downloading program components (part 1 of 2)..."
printf '    '
( download "$PYREL/cpython-$PYVER-$PYT-install_only.tar.gz" "$WORK/py.tgz" >> "$LOG" 2>&1 ) & spin $! \
  || fail "Some program components could not be downloaded. Check the internet connection, then run this installer again."
rm -rf "$DEST"
mkdir -p "$ROOT" || fail "Cannot write to $ROOT"
tar xzf "$WORK/py.tgz" -C "$ROOT" >> "$LOG" 2>&1 || fail "The downloaded components are damaged. Please run the installer again."
PY="$ROOT/python/bin/python3"
[ -x "$PY" ] || fail "The downloaded components are incomplete. Please run the installer again."

# ---------------------------------------------------------------- packages (part 2)
step "Downloading program components (part 2 of 2, about 200 MB - this takes a few minutes)..."
printf '    '
( "$PY" -E -s -m pip install --disable-pip-version-check --no-warn-script-location --no-cache-dir \
      --only-binary=:all: --retries 5 --timeout 60 -q -r "$WORK/$REQ" >> "$LOG" 2>&1 ) & spin $! \
  || fail "Some program components could not be downloaded. Check the internet connection, then run this installer again."
SITE="$("$PY" -E -s -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ "$OS" = "Linux" ] && [ -d "$WORK/xlibs" ]; then
  cp "$WORK"/xlibs/* "$SITE/PySide6/Qt/lib/" 2>> "$LOG"    # X11 libraries Qt needs (missing on fresh Ubuntu)
fi
rm -rf "$ROOT/python/share" "$ROOT"/python/lib/python3.12/test "$ROOT"/python/lib/python3.12/idlelib 2>/dev/null
"$PY" -E -s "$WORK/slim.py" "$SITE" >> "$LOG" 2>&1          # drop test suites / developer files (~175 MB)
fi

# ---------------------------------------------------------------- the application
step "Installing $APP..."
cp -R "$WORK/app" "$ROOT/app" || fail "Cannot copy the program files."
step "Preparing $APP for its first start (about a minute)..."
printf '    '
( "$PY" -E -s -m compileall -q -j 0 "$ROOT/python/lib" "$ROOT/app" >> "$LOG" 2>&1 ) & spin $!

# ---------------------------------------------------------------- shortcuts
if [ "$OS" = "Darwin" ]; then
  C="$DEST/Contents"
  mkdir -p "$C/MacOS"
  cp "$WORK/app.icns" "$ROOT/app.icns"
  cp "$ROOT/python/bin/python3.12" "$C/MacOS/$APP-bin"       # the app's own process -> menu bar + Dock say BehavioralTracker
  ln -s "Resources/python/lib" "$C/lib"                        # libpython next to it (@executable_path/../lib)
  cat > "$C/MacOS/$APP" <<LAUNCH
#!/bin/bash
C="\$(cd "\$(dirname "\$0")/.." && pwd)"
unset PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONEXECUTABLE VIRTUAL_ENV
export PYTHONHOME="\$C/Resources/python" PYTHONNOUSERSITE=1
export BT_START_DIR="\$PWD"
cd "\$C/Resources/app"
exec "\$C/MacOS/$APP-bin" -s "\$C/Resources/app/main_pro.py" "\$@"
LAUNCH
  chmod +x "$C/MacOS/$APP"
  cat > "$C/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>$APP</string>
  <key>CFBundleDisplayName</key><string>$APP</string>
  <key>CFBundleIdentifier</key><string>com.manojkumarmunusamy.behavioraltracker</string>
  <key>CFBundleExecutable</key><string>$APP</string>
  <key>CFBundleIconFile</key><string>app.icns</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VER</string>
  <key>CFBundleVersion</key><string>$VER</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>$APP records experiment videos from a camera.</string>
</dict></plist>
PLIST
  touch "$DEST"
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$DEST" >/dev/null 2>&1
else
  cp "$WORK/behavioraltracker.png" "$ROOT/behavioraltracker.png"
  cat > "$ROOT/$APP" <<LAUNCH
#!/bin/bash
D="\$(cd "\$(dirname "\$(readlink -f "\$0")")" && pwd)"
export BT_START_DIR="\$PWD"
cd "\$D/app"
exec "\$D/python/bin/python3" -E -s "\$D/app/main_pro.py" "\$@"
LAUNCH
  chmod +x "$ROOT/$APP"
  mkdir -p "$HOME/.local/share/applications"
  DESK="$HOME/.local/share/applications/behavioraltracker.desktop"
  cat > "$DESK" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$APP
Comment=Animal behaviour tracking and analysis
Exec="$ROOT/$APP"
Icon=$ROOT/behavioraltracker.png
Terminal=false
Categories=Science;Education;
StartupWMClass=$APP
DESKTOP
  cat > "$ROOT/uninstall.sh" <<UNINST
#!/bin/bash
rm -rf "$DEST" "$DESK" "\$HOME/Desktop/behavioraltracker.desktop"
printf 'Also delete your settings, history and licence (~/.behavioraltracker)? [y/N] '
read -r a; case "\$a" in y|Y*) rm -rf "\$HOME/.behavioraltracker";; esac
echo "$APP removed. Your videos and results folders were not touched."
UNINST
  chmod +x "$ROOT/uninstall.sh"
  update-desktop-database "$HOME/.local/share/applications" >/dev/null 2>&1
  if [ -d "$HOME/Desktop" ] && [ -t 0 ]; then
    printf '\n  Add a shortcut on the Desktop? [y/N] '; read -r a < /dev/tty
    case "$a" in
      y|Y*) cp "$DESK" "$HOME/Desktop/"; chmod +x "$HOME/Desktop/behavioraltracker.desktop"
            gio set "$HOME/Desktop/behavioraltracker.desktop" metadata::trusted true >/dev/null 2>&1 ;;
    esac
  fi
fi

say ""
say "  ============================================================"
say "     $APP is installed."
if [ "$OS" = "Darwin" ]; then
  say "     Open it from Launchpad, Spotlight or the Applications folder"
  say "     in your home folder (~/Applications)."
else
  say "     Open it from the applications menu."
  say "     To remove it later:  $ROOT/uninstall.sh"
fi
[ "$EDITION" = "licensed" ] && say "     On first start it shows your Machine ID - email it to $SUPPORT"
say "  ============================================================"
if [ -t 0 ] || [ "$OS" = "Darwin" ]; then
  printf '\n  Start %s now? [Y/n] ' "$APP"; read -r a < /dev/tty 2>/dev/null
  case "$a" in n|N*) ;; *)
    if [ "$OS" = "Darwin" ]; then open "$DEST"; else (nohup "$ROOT/$APP" >/dev/null 2>&1 &); fi ;;
  esac
fi
exit 0
__BT_PAYLOAD_BELOW__
