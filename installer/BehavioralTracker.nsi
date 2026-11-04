; BehavioralTracker 1.0.0 - Windows Setup
; Built by installer/build_installer.sh (makensis).  The staged folder holds:
;   BehavioralTracker.exe   small launcher (icon + version info)
;   python\                 private Python 3.12 with every package the app needs
;   app\                    the application (main_pro.py, pro_ui, tracking, analysis, output, resources)
;   Sample video\           dummy_behavior_test.mp4
; Installs for the current user (no administrator rights needed).
; -DONLINE : python\ holds only Python + pip; the packages (app\requirements-win.txt) are
;            downloaded from PyPI during the installation (small Setup file).

Unicode true
Target amd64-unicode
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 128

!ifndef STAGE
  !define STAGE "stage\BehavioralTracker"
!endif
!ifndef ASSETS
  !define ASSETS "."
!endif
!ifndef ESTSIZE
  !define ESTSIZE 600000
!endif
!ifndef OUTFILE
  !define OUTFILE "BehavioralTracker_Setup_1.0.0.exe"
!endif

!define APPNAME "BehavioralTracker"
!define APPVERSION "1.0.0"
!define PUBLISHER "Manojkumar Munusamy"
!define SUPPORT "manojkumarmunusamy1101@outlook.com"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"

Name "${APPNAME} ${APPVERSION}"
Caption "${APPNAME} ${APPVERSION} Setup"
BrandingText "${APPNAME} ${APPVERSION}"
OutFile "${OUTFILE}"
!ifdef SIGNCMD
!uninstfinalize '${SIGNCMD}'    ; code signing of the uninstaller (build with SIGN_PFX)
!endif
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
RequestExecutionLevel user
ShowInstDetails nevershow
ShowUninstDetails nevershow

VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "ProductVersion" "${APPVERSION}"
VIAddVersionKey "FileVersion" "${APPVERSION}"
VIAddVersionKey "FileDescription" "${APPNAME} ${APPVERSION} Setup"
VIAddVersionKey "CompanyName" "${PUBLISHER}"
VIAddVersionKey "LegalCopyright" "${PUBLISHER}"
VIAddVersionKey "Comments" "Support: ${SUPPORT}"

!include "MUI2.nsh"
!include "LogicLib.nsh"

!define MUI_ICON "${ASSETS}\app.ico"
!define MUI_UNICON "${ASSETS}\app.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${ASSETS}\welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${ASSETS}\welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${ASSETS}\header.bmp"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Welcome to ${APPNAME} ${APPVERSION}"
!ifdef ONLINE
  !define MUI_WELCOMEPAGE_TEXT "This will install ${APPNAME} - animal behaviour tracking and analysis (open field, EPM, Y / T maze, light-dark box, NOR, social interaction).$\r$\n$\r$\nSome program components are downloaded during the installation - keep the computer connected to the internet.$\r$\n$\r$\nClick Next to continue."
!else
  !define MUI_WELCOMEPAGE_TEXT "This will install ${APPNAME} - animal behaviour tracking and analysis (open field, EPM, Y / T maze, light-dark box, NOR, social interaction).$\r$\n$\r$\nEverything the program needs is included - nothing else has to be installed and no internet connection is needed.$\r$\n$\r$\nClick Next to continue."
!endif
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!define MUI_COMPONENTSPAGE_TEXT_TOP "Choose the shortcuts you want. ${APPNAME} itself is always installed."
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "${APPNAME} is installed"
!define MUI_FINISHPAGE_TEXT "${APPNAME} ${APPVERSION} has been installed on your computer.$\r$\n$\r$\nA sample video is in the 'Sample video' folder of the installation."
!define MUI_FINISHPAGE_RUN "$INSTDIR\BehavioralTracker.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APPNAME} now"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ----------------------------------------------------------------------------- progress helpers
!macro Status TEXT
  SetDetailsPrint textonly
  DetailPrint "${TEXT}"
  SetDetailsPrint none
!macroend

; while a long step runs, the progress bar keeps moving ("marquee") instead of standing still
!ifndef PBS_MARQUEE
  !define PBS_MARQUEE 0x08
!endif
!ifndef PBM_SETMARQUEE
  !define PBM_SETMARQUEE 0x40A
!endif
Function MarqueeOn
  FindWindow $1 "#32770" "" $HWNDPARENT
  GetDlgItem $1 $1 1004
  System::Call 'user32::GetWindowLongW(p r1, i -16) i .r2'
  IntOp $2 $2 | ${PBS_MARQUEE}
  System::Call 'user32::SetWindowLongW(p r1, i -16, i r2) i .r3'
  SendMessage $1 ${PBM_SETMARQUEE} 1 30
FunctionEnd

Function MarqueeOff
  FindWindow $1 "#32770" "" $HWNDPARENT
  GetDlgItem $1 $1 1004
  SendMessage $1 ${PBM_SETMARQUEE} 0 0
  System::Call 'user32::GetWindowLongW(p r1, i -16) i .r2'
  IntOp $3 ${PBS_MARQUEE} ~
  IntOp $2 $2 & $3
  System::Call 'user32::SetWindowLongW(p r1, i -16, i r2) i .r3'
FunctionEnd

; ----------------------------------------------------------------------------- sections
Section "${APPNAME}" SecApp
  SectionIn RO
  ; only short, friendly status lines are shown -- never file or component names
  SetDetailsPrint none
  ; replace an older installation cleanly (the user's settings / results are elsewhere and are kept)
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"

  SetOutPath "$INSTDIR"
  !insertmacro Status "Installing ${APPNAME}..."
  File "${STAGE}\BehavioralTracker.exe"
  File "${STAGE}\app.ico"
  !insertmacro Status "Installing program files..."
  File /r "${STAGE}\python"
  File /r "${STAGE}\app"

!ifdef ONLINE
  !insertmacro Status "Downloading program components - this takes a few minutes..."
  Call MarqueeOn
  retry_packages:
  nsExec::Exec '"$INSTDIR\python\python.exe" -E -s -m pip install --quiet --no-warn-script-location --disable-pip-version-check --progress-bar off --retries 5 --timeout 60 --only-binary=:all: -r "$INSTDIR\app\requirements-win.txt"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "Some program components could not be downloaded.$\r$\n$\r$\nCheck that this computer is connected to the internet, then press Retry." IDRETRY retry_packages
    Call MarqueeOff
    RMDir /r "$INSTDIR\python"
    RMDir /r "$INSTDIR\app"
    Delete "$INSTDIR\BehavioralTracker.exe"
    Delete "$INSTDIR\app.ico"
    RMDir "$INSTDIR"
    Abort "Installation cancelled."
  ${EndIf}
  Delete "$INSTDIR\python\Lib\site-packages\*.whl"
  !insertmacro Status "Preparing ${APPNAME} for its first start..."
  nsExec::Exec '"$INSTDIR\python\python.exe" -E -s -m compileall -q "$INSTDIR\app"'
  Pop $0
  Call MarqueeOff
!else
  !insertmacro Status "Preparing ${APPNAME} for its first start (this can take a minute)..."
  Call MarqueeOn
  nsExec::Exec '"$INSTDIR\python\python.exe" -E -s -m compileall -q -j 0 "$INSTDIR\python\Lib" "$INSTDIR\app"'
  Pop $0
  Call MarqueeOff
!endif

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${APPVERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "${UNINSTKEY}" "Contact" "${SUPPORT}"
  WriteRegStr HKCU "${UNINSTKEY}" "HelpLink" "mailto:${SUPPORT}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\BehavioralTracker.exe"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "EstimatedSize" ${ESTSIZE}
  ; double-click a BT file in Explorer -> it opens in BehavioralTracker
  WriteRegStr HKCU "Software\Classes\.bta" "" "BehavioralTracker.Analysis"
  WriteRegStr HKCU "Software\Classes\.bta\OpenWithProgids" "BehavioralTracker.Analysis" ""
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Analysis" "" "BT Analysis file"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Analysis\DefaultIcon" "" "$INSTDIR\BehavioralTracker.exe,0"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Analysis\shell\open\command" "" '"$INSTDIR\BehavioralTracker.exe" "%1"'
  WriteRegStr HKCU "Software\Classes\.btms" "" "BehavioralTracker.Scoring"
  WriteRegStr HKCU "Software\Classes\.btms\OpenWithProgids" "BehavioralTracker.Scoring" ""
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Scoring" "" "BT Manual Scoring file"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Scoring\DefaultIcon" "" "$INSTDIR\BehavioralTracker.exe,0"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Scoring\shell\open\command" "" '"$INSTDIR\BehavioralTracker.exe" "%1"'
  WriteRegStr HKCU "Software\Classes\.btf" "" "BehavioralTracker.Freezing"
  WriteRegStr HKCU "Software\Classes\.btf\OpenWithProgids" "BehavioralTracker.Freezing" ""
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Freezing" "" "BT Freezing file"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Freezing\DefaultIcon" "" "$INSTDIR\BehavioralTracker.exe,0"
  WriteRegStr HKCU "Software\Classes\BehavioralTracker.Freezing\shell\open\command" "" '"$INSTDIR\BehavioralTracker.exe" "%1"'
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
  !insertmacro Status "Done."
SectionEnd

Section "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\BehavioralTracker.exe" "" "$INSTDIR\BehavioralTracker.exe" 0 SW_SHOWNORMAL "" "Animal behaviour tracking and analysis"
SectionEnd

Section "Start menu shortcut" SecStart
  CreateShortcut "$SMPROGRAMS\${APPNAME}.lnk" "$INSTDIR\BehavioralTracker.exe" "" "$INSTDIR\BehavioralTracker.exe" 0 SW_SHOWNORMAL "" "Animal behaviour tracking and analysis"
SectionEnd

Section "Sample video" SecSample
  SetOutPath "$INSTDIR\Sample video"
  SetDetailsPrint none
  File /nonfatal "${STAGE}\Sample video\*.mp4"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecApp} "The ${APPNAME} program."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Put a ${APPNAME} shortcut on the Desktop."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecStart} "Add ${APPNAME} to the Start menu."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecSample} "A short test video to try the program."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ----------------------------------------------------------------------------- uninstall
Section "Uninstall"
  Delete "$DESKTOP\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}.lnk"
  SetDetailsPrint textonly
  DetailPrint "Removing ${APPNAME}..."
  SetDetailsPrint none
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\Sample video"
  Delete "$INSTDIR\BehavioralTracker.exe"
  Delete "$INSTDIR\app.ico"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\${APPNAME}"
  DeleteRegKey HKCU "Software\Classes\BehavioralTracker.Analysis"
  DeleteRegKey HKCU "Software\Classes\BehavioralTracker.Scoring"
  DeleteRegKey HKCU "Software\Classes\.bta"
  DeleteRegKey HKCU "Software\Classes\.btms"
  DeleteRegKey HKCU "Software\Classes\.btf"
  DeleteRegKey HKCU "Software\Classes\BehavioralTracker.Freezing"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
  ; settings + experiment history (videos and result folders are never touched)
  MessageBox MB_YESNO|MB_ICONQUESTION "Also delete your ${APPNAME} settings and experiment history?$\r$\n$\r$\nChoose Yes for a completely fresh start next time.$\r$\n(Your videos and result folders are never deleted.)" /SD IDNO IDNO keep_user_data
  RMDir /r "$APPDATA\${APPNAME}"
  RMDir /r "$LOCALAPPDATA\${APPNAME}"
  keep_user_data:
SectionEnd
