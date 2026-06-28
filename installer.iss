; =============================================================================
; Inno Setup Script for RadVision-Fusion App
; =============================================================================

[Setup]
AppName=RadVision-Fusion
AppVersion=1.0
AppPublisher=IamOumarIbrahim
DefaultDirName={userpf}\RadVision-Fusion
DefaultGroupName=RadVision-Fusion
OutputDir=.
OutputBaseFilename=RadVision-Fusion-Setup
Compression=lzma
SolidCompression=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest

[Files]
; Copy all project files into the installation directory
Source: "radvision_fusion\*"; DestDir: "{app}\radvision_fusion"; Flags: recursesubdirs createallsubdirs
Source: "radvision_fusion\requirements.txt"; DestDir: "{app}"

[Icons]
; Shortcut to run the Streamlit web dashboard locally using user's python environment
Name: "{group}\RadVision-Fusion Dashboard"; Filename: "powershell.exe"; Parameters: "-NoExit -Command ""cd '{app}'; pip install -r requirements.txt; python -m streamlit run radvision_fusion/app.py"""
Name: "{userdesktop}\RadVision-Fusion Dashboard"; Filename: "powershell.exe"; Parameters: "-NoExit -Command ""cd '{app}'; pip install -r requirements.txt; python -m streamlit run radvision_fusion/app.py"""
