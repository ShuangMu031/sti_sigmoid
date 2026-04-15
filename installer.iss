[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName=Eagle Eye Stitcher
AppVersion=1.2.0
AppPublisher=Sigmoid
AppPublisherURL=https://github.com/ShuangMu031/sti_sigmoid
AppSupportURL=https://github.com/ShuangMu031/sti_sigmoid
DefaultDirName={autopf}\Eagle Eye Stitcher
DefaultGroupName=Eagle Eye Stitcher
AllowNoIcons=yes
OutputDir=installer
OutputBaseFilename=Eagle_Eye_Stitcher_Setup
SetupIconFile=assets\icons\cat.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\Eagle Eye Stitcher.exe
UninstallDisplayName=Eagle Eye Stitcher

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Eagle Eye Stitcher\Eagle Eye Stitcher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Eagle Eye Stitcher\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Eagle Eye Stitcher"; Filename: "{app}\Eagle Eye Stitcher.exe"
Name: "{group}\{cm:UninstallProgram,Eagle Eye Stitcher}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Eagle Eye Stitcher"; Filename: "{app}\Eagle Eye Stitcher.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Eagle Eye Stitcher.exe"; Description: "{cm:LaunchProgram,Eagle Eye Stitcher}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
