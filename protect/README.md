# BehavioralTracker — protection & licensing kit

This folder turns the normal app into a **protected** build and lets you issue
**node-locked licences**.  Everything here is yours; none of it ships to customers
except the obfuscated bundle the build produces.

> **One thing I did not build, on purpose:** an "auto-destroy / delete the source
> if someone tries to reverse-engineer" trap.  A tamper trap that *deletes files*
> fires on false alarms too (a customer's antivirus sandbox, a VM, a debugger
> attached by their IT) and would wipe an honest customer's install — and a
> "detect analysis → destroy data" mechanism is not something I'll ship.
> Instead the build **fails closed**: if any file is modified, the app simply
> **refuses to start** ("integrity check failed — reinstall").  Nothing is deleted.
> That protects your code without the risk of trashing a real user's machine.

No client-side protection is truly unbreakable (a determined expert with the
running program can always dump memory — this is the same limit every DRM has).
What this kit does is make casual reverse-engineering and copying **much** harder,
and tie each copy to one machine.

---

## The 5 layers

1. **No plain source ships** — every app module (`pro_ui`, `tracking`, `analysis`,
   `output`, the licence code) is shipped only as an **AES-encrypted blob** (`.btx`).
2. **Encrypted UI** — the Qt `.ui` is encrypted too and decrypted **in memory**;
   it never sits on disk in readable form.
3. **Compiled, key-split loader** — the decryptor is bytecode only (`loader.pyc`),
   and the AES key is broken into masked pieces, not a plain string.
4. **Integrity lock** — every blob is hashed and the manifest is HMAC-signed with
   the same key; edit or replace *anything* and the app refuses to start (fail-closed).
5. **Node-locked licence** (licensed build) — an **Ed25519-signed** licence bound to
   one computer's Machine ID.  The app holds only the *public* key, so no one can
   forge a licence; only you hold the private key.

The only plain files in a protected build are `main_pro.py` (a ~40-line bootstrap
that just turns on the loader) and an empty `_sec/__init__.py`.

---

## Build the installers

Needs (on Ubuntu / WSL): `sudo apt-get install -y nsis mingw-w64 python3.12 python3-pip`
and `python3 -m pip install cryptography` plus the cross wheels pull automatically.

```bash
# both protected editions, online + offline Setups
KINDS="unlicensed licensed" bash installer/build_installer.sh both

# just the small online installers (download packages while installing)
KINDS="unlicensed licensed" bash installer/build_installer.sh online

# the old unprotected build (plain source visible) if you ever need it
KINDS="plain" bash installer/build_installer.sh both
```

Output in `installer/Output/`:

| File | Edition |
|---|---|
| `BehavioralTracker_Setup_1.0.0_unlicensed.exe` | protected, runs on any PC (online) |
| `BehavioralTracker_Setup_1.0.0_unlicensed_offline.exe` | same, no internet needed |
| `BehavioralTracker_Setup_1.0.0_licensed.exe` | protected + needs a licence (online) |
| `BehavioralTracker_Setup_1.0.0_licensed_offline.exe` | same, no internet needed |

The build compiles the modules with **Python 3.12** (`python3.12`) so the bytecode
matches the Python 3.12 that ships inside the app — this is required, don't use 3.13.

---

## Licensing — how it works

```
┌─────────┐  1. app shows Machine ID   ┌──────────┐
│ customer │ ─────────────────────────▶ │   you    │
│          │     (emails it to you)     │          │
│          │  2. you send back .key     │ keygen   │
│          │ ◀───────────────────────── │ (private │
│          │     3. loads it in the app │  key)    │
└─────────┘                            └──────────┘
```

### One-time setup (do this once, on YOUR computer)

```bash
python protect/keygen.py init        # creates protect/keys/private.pem + public.pem, asks for a password
```

- `private.pem` is **secret** — never share it, keep a safe backup.  It is the only
  thing that can sign licences.
- It is stored **encrypted with your password** (AES, PKCS#8): a copy of the
  keys folder is useless without the password.  Every licence asks for it; it is never
  written anywhere.  **Forgotten password = no new licences** (licences already issued keep
  working) — write it down and keep it safe.
- A key made by an older licence generator has no password: the licence window asks you to **create one**
  the first time you open it (or run `python protect/keygen.py set-password`).  Change it
  later with `set-password` or the **Change password…** button.  The key pair stays the
  same, so every licence you already issued keeps working.  Afterwards delete every other
  copy of the old keys folder (old zip files, USB sticks) — those copies have no password.
- `public.pem` is copied into the app automatically when you build the *licensed*
  edition.

> A ready-made keypair is already in `protect/keys/` so the licensed installer in
> this delivery works out of the box.  **For real security, run
> `python protect/keygen.py init --force` on your own machine to make a fresh
> private key that never left your computer, then rebuild the licensed installer.**

### Issue a licence for a customer

The customer installs the *licensed* app; on first run it shows their **Machine ID**
(e.g. `BT-1A2B-3C4D-5E6F-7G8H`) and your email `manojkumarmunusamy1101@outlook.com`.
They email you the Machine ID.  You run:

```bash
python protect/keygen.py issue --machine-id BT-1A2B-3C4D-5E6F-7G8H --name "Dr X lab"      # asks for your password
# optional time-limited licence:
python protect/keygen.py issue --machine-id BT-... --name "Trial" --days 30
# or a window instead of the command line:
python protect/keygen.py gui
```

That writes `protect/issued/BT-….key`.  Email it back; the customer clicks
**"I have a licence — load it…"** in the app.  The licence is saved to
`%APPDATA%\BehavioralTracker\license.key` and the app runs on that computer only.

- The Machine ID is a one-way hash of hardware IDs — it carries no personal data and
  is safe to email.
- A licence works **only** on the computer whose Machine ID it was issued for.
- Expired or edited licences are rejected; the activation window reappears.

### Why not tie it to an IP or a login?
An IP address changes (Wi-Fi, mobile, VPN) so the licence would keep breaking, and a
"login" would need a server you run 24/7.  A hardware Machine ID is stable and works
fully offline — better for desktop lab software.

---

## Files here

| File | What it is | Ships to customers? |
|---|---|---|
| `build_secure.py` | turns the app into the encrypted bundle | no |
| `licensing.py` | machine-ID + licence verify (gets encrypted into the app) | as a blob only |
| `keygen.py` | **your** licence generator (uses the private key) | **NEVER** |
| `keys/private.pem` | **your secret signing key** | **NEVER** |
| `keys/public.pem` | public key (baked into licensed builds) | inside the app |
| `issued/*.key` | licences you generated | each to its own customer |
