; TFLiteTraining Windows Installer
; Build with Inno Setup 6:  ISCC.exe installer.iss
; (install Inno Setup from https://jrsoftware.org/isinfo.php, or: winget install JRSoftware.InnoSetup)

#define AppName "TFLiteTraining"
#ifndef AppVersion
  #define AppVersion "2.4.14"
#endif
#define AppPublisher "TFLiteTraining"
#define AppExeName "TFLiteTraining.exe"

[Setup]
AppId={{371211F1-9C88-4C54-93E8-AD6782994AF5}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=TFLiteTraining-Setup-v{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "dist\TFLiteTraining\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
