"""Node-locked licensing for BehavioralTracker (runtime side).

This module is part of the protected application (it is obfuscated into the
secure bundle like every other app module).  It knows only the PUBLIC key, so
it can *verify* a licence but never *create* one -- licences are signed by the
developer's private key, which lives only in protect/keygen.py and never ships.

A licence is a base64 text blob:  base64( json({"p": payload, "s": base64(ed25519_sig)}) )
payload = {"machine_id": "...", "licensee": "...", "issued": "...",
           "expires": "YYYY-MM-DD" | null, "product": "BehavioralTracker"}
The signature covers the canonical JSON of the payload, so changing any field
(for example the expiry date) makes the licence invalid.

verify_license() returns (ok: bool, info_or_reason: dict|str).
"""
import base64
import hashlib
import json
import os
import platform
import sys
import uuid


# --------------------------------------------------------------------------- machine fingerprint
def _windows_machine_guid():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as k:
            return winreg.QueryValueEx(k, "MachineGuid")[0]
    except Exception:
        return ""


def _windows_volume_serial():
    try:
        import ctypes
        vol = ctypes.c_uint(0)
        drive = os.environ.get("SystemDrive", "C:") + "\\"
        ctypes.windll.kernel32.GetVolumeInformationW(ctypes.c_wchar_p(drive), None, 0, ctypes.byref(vol),
                                                     None, None, None, 0)
        return str(vol.value)
    except Exception:
        return ""


def _linux_machine_id():
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(p) as f:
                v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def _mac_platform_uuid():
    """The Mac's hardware UUID (IOPlatformUUID) -- stable across OS reinstalls."""
    try:
        import subprocess
        out = subprocess.run(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "IOPlatformUUID" in line:
                return line.split("=")[-1].strip().strip('"')
    except Exception:
        pass
    return ""


def _raw_fingerprint():
    parts = [platform.system(), platform.machine()]
    if sys.platform == "win32":
        parts += [_windows_machine_guid(), _windows_volume_serial()]
    elif sys.platform == "darwin":
        parts += [_mac_platform_uuid()]
    else:
        parts += [_linux_machine_id()]
    if not any(parts[2:]):                       # last-resort: MAC address
        parts.append(f"{uuid.getnode():012x}")
    return "|".join(parts)


def machine_id():
    """Short, stable, human-readable ID for this computer (e.g. BT-1A2B-3C4D-5E6F).

    The same computer always produces the same ID; different computers almost
    never collide.  It contains no personal data -- it is a one-way hash of
    hardware identifiers, safe to email."""
    digest = hashlib.sha256(_raw_fingerprint().encode("utf-8", "replace")).hexdigest().upper()
    return "BT-" + "-".join(digest[i:i + 4] for i in (0, 4, 8, 12))


# --------------------------------------------------------------------------- licence verification
def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_license_bytes(blob, public_key_pem, *, product="BehavioralTracker", today=None):
    """(ok, info|reason).  `blob` is the licence file's bytes, `public_key_pem`
    the embedded Ed25519 public key (PEM bytes)."""
    import datetime
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        env = json.loads(base64.b64decode(blob.strip(), validate=False).decode("utf-8"))
        payload = env["p"]
        sig = base64.b64decode(env["s"])
    except Exception:
        return False, "This licence file is not readable. Please request a new one."
    try:
        load_pem_public_key(public_key_pem).verify(sig, _canonical(payload))
    except InvalidSignature:
        return False, "This licence is not valid for this product (signature check failed)."
    except Exception:
        return False, "This licence could not be checked. Please request a new one."
    if payload.get("product") not in (product, None):
        return False, "This licence is for a different product."
    if payload.get("machine_id") != machine_id():
        return False, "This licence belongs to a different computer."
    exp = payload.get("expires")
    if exp:
        try:
            d = datetime.date.fromisoformat(exp)
            now = today or datetime.date.today()
            if now > d:
                return False, f"This licence expired on {exp}."
        except ValueError:
            return False, "This licence has an invalid expiry date."
    return True, payload
