App icon, generated from the image you provided:

- icon.png  -- used by the running app itself (window/taskbar icon, all OSes)
- icon.ico  -- Windows .exe file icon (PyInstaller --icon)
- icon.icns -- macOS .app bundle icon (PyInstaller --icon)

All three are bundled into the packaged app via --add-data in build.yml.
