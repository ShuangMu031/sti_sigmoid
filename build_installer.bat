@echo off
echo ========================================
echo   Eagle Eye Stitcher - Inno Setup Compiler
echo ========================================
echo.

cd /d "%~dp0"

set ISCC=""
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
if exist "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" set ISCC="C:\Program Files (x86)\Inno Setup 5\ISCC.exe"
if exist "C:\Program Files\Inno Setup 5\ISCC.exe" set ISCC="C:\Program Files\Inno Setup 5\ISCC.exe"

if %ISCC%=="" (
    echo [INFO] Inno Setup Compiler not found
    echo.
    echo Please do manually:
    echo 1. Open Inno Setup Compiler
    echo 2. Open file: %cd%\installer.iss
    echo 3. Click Build -^> Compile
    echo.
    echo Or double-click installer.iss
    pause
    exit /b 0
)

echo [1/2] Compiling installer...
%ISCC% installer.iss

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Compilation complete!
echo.
echo Output: installer\Eagle_Eye_Stitcher_Setup.exe
echo ========================================
pause
