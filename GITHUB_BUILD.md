# Build BehavioralTracker 1.0.0 on GitHub — licensed edition

This folder builds the **online installers** on GitHub (nothing offline is made or stored):

| File | For | While installing |
|---|---|---|
| `BehavioralTracker_Setup_1.0.0_licensed.exe` | Windows 10 / 11 | downloads the Python packages (~220 MB) |
| `BehavioralTracker_Setup_1.0.0_linux_licensed.run` | Ubuntu 22.04+, Debian 12+, Fedora 35+ | downloads Python + packages (~250 MB) |
| `BehavioralTracker_Setup_1.0.0_mac_licensed.zip` | macOS 13+ (Apple Silicon, Intel) | downloads Python + packages (~250 MB) |

The computer needs the internet **during installation**; after that BehavioralTracker works offline.
This edition **asks for a licence** on every computer: the app shows the Machine ID; the licences you make with your LicenceTools work (the same public key, `protect/keys/public.pem`, is inside).

## 1. Make a PRIVATE repository
github.com → **New repository** → choose **Private**. Do not use a public repository (and not the public
`Mano1101/BehavioralTracker`): anyone could read and copy the program code.

## 2. Put this folder in it
Unzip, then upload **everything inside `BehavioralTracker_github`**, including the hidden `.github` folder and
`.gitignore`. Easiest with GitHub Desktop, or in a terminal inside the folder:

    git init
    git add .
    git commit -m "BehavioralTracker 1.0.0"
    git branch -M main
    git remote add origin https://github.com/<your-account>/<your-repo>.git
    git push -u origin main

(The browser upload can skip hidden folders. If **Actions** shows no workflow afterwards: Add file → Create new
file → name it `.github/workflows/build-installers.yml` and paste that file's text.)

## 3. Build
**Actions** tab → **Build installers (online)** → **Run workflow**. It takes about 15–25 minutes and ends with a
test: the Linux installer is installed on GitHub's computer (downloading its packages) and the app is started.

## 4. Download
Open the finished (green) run → **Artifacts** → **BehavioralTracker-installers-licensed** (a zip with the three
installers). Artifacts are kept for 30 days.

For a download page: push a tag — `git tag v1.0.0` then `git push origin v1.0.0` — and the installers
also appear under **Releases** of the repository (private: only people you invite can open it).

## Never put these in the repository
The licence generator (`keygen.py`, `Generate_Licence.bat`), your private key (`keys/private.pem`) and issued
licences (`*.key`). Keep them only on your own computer. The `.gitignore` here blocks them if they are copied in
by mistake. The build needs no secret.

## Tamil (surukkam)
1. GitHub-la oru **Private** repository create pannunga (public vendaam — code ellaarukkum theriyum).
2. Indha folder-la irukkura ellaa files-ayum (`.github` folder serthu) andha repository-la upload / push pannunga.
3. **Actions → Build installers (online) → Run workflow** press pannunga (~20 nimisham).
4. Green aanadhum **Artifacts**-la installers download pannunga. Ivai online installers — install pannumbodhu
   internet venum.
5. LicenceTools (keygen, private key) eppovum GitHub-la podaadheenga.
