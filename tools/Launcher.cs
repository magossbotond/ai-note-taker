// Tiny launcher for AI Interview Notetaker: starts main.py with the project's
// .venv Python, with no console window. Stdout/stderr go to app.log, and if the
// app exits with an error the tail of that log is shown in a message box.
// Build with tools\build_launcher.bat (uses the C# compiler that ships with Windows).
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

static class Launcher
{
    const string Title = "AI Interview Notetaker";

    [STAThread]
    static int Main()
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(dir, ".venv", "Scripts", "python.exe");
        string mainPy = Path.Combine(dir, "main.py");
        string log = Path.Combine(dir, "app.log");

        if (!File.Exists(python) || !File.Exists(mainPy))
        {
            MessageBox.Show(
                "Could not find .venv\\Scripts\\python.exe or main.py next to this program.\n\n" +
                "Run setup.bat first, and keep this exe in the project folder.",
                Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        // Go through cmd.exe so output redirection to app.log is handled for us.
        var psi = new ProcessStartInfo("cmd.exe",
            "/c \"\"" + python + "\" \"" + mainPy + "\" > \"" + log + "\" 2>&1\"")
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"; // Hungarian text in logs
        psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

        using (var proc = Process.Start(psi))
        {
            proc.WaitForExit();
            if (proc.ExitCode != 0)
            {
                string tail = "(no log output)";
                try
                {
                    tail = string.Join("\n", File.ReadAllLines(log).Reverse().Take(15).Reverse());
                }
                catch { }
                MessageBox.Show(
                    "The app exited with code " + proc.ExitCode + ".\n\nLast log lines (full log: app.log):\n\n" + tail,
                    Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return proc.ExitCode;
        }
    }
}
