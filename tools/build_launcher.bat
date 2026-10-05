@echo off
REM Compiles tools\Launcher.cs into "AI Interview Notetaker.exe" in the project root,
REM using the C# compiler built into Windows (no extra tools needed).
setlocal
set "ROOT=%~dp0.."
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
if not exist "%CSC%" (
    echo C# compiler not found - .NET Framework 4 is required.
    exit /b 1
)
"%CSC%" /nologo /target:winexe /optimize+ /r:System.Windows.Forms.dll ^
    /out:"%ROOT%\AI Interview Notetaker.exe" "%~dp0Launcher.cs"
if errorlevel 1 exit /b 1
echo       Built "AI Interview Notetaker.exe"
exit /b 0
