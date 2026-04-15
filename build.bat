@echo off
echo ========================================
echo   Eagle Eye Stitcher - Build Script
echo ========================================
echo.

cd /d "%~dp0"

if not exist "assets\icons\cat.ico" (
    echo [ERROR] Icon file not found
    pause
    exit /b 1
)

echo [1/3] Cleaning old build files...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo [2/3] Building...
.venv_build\Scripts\pyinstaller.exe build.spec --clean

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.
echo Output: dist\Eagle Eye Stitcher\Eagle Eye Stitcher.exe
echo ========================================
pause
