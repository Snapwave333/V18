using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

class VibesLauncher
{
    static void Main()
    {
        Console.Title = "Vibes V18 Launcher";
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══════════════════════════════════════");
        Console.WriteLine("         VIBES V18 — Launching...     ");
        Console.WriteLine("═══════════════════════════════════════");
        Console.ResetColor();

        string baseDir = @"C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine";
        string pythonDir = Path.Combine(baseDir, "python");
        string pythonExe = Path.Combine(pythonDir, @".venv\Scripts\python.exe");
        string ipcDir = Path.Combine(pythonDir, "temp_ipc");

        // Verify python exists
        if (!File.Exists(pythonExe))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("ERROR: Python not found at: " + pythonExe);
            Console.WriteLine("Press any key to exit...");
            Console.ResetColor();
            Console.ReadKey();
            return;
        }

        // Kill stale python processes
        Console.WriteLine("[1/3] Killing stale processes...");
        foreach (var proc in Process.GetProcessesByName("python"))
        {
            try { proc.Kill(); } catch { }
        }
        Thread.Sleep(500);

        // Purge old frames for a fresh start
        Console.WriteLine("[2/3] Purging old frames...");
        if (Directory.Exists(ipcDir))
        {
            try { Directory.Delete(ipcDir, true); } catch { }
        }

        // Launch VJ Engine
        Console.WriteLine("[3/3] Launching VJ Engine...");
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = "-u main.py --worker deform --delay 0",
                WorkingDirectory = pythonDir,
                UseShellExecute = false
            };
            Process.Start(psi);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("VJ Engine started! This window will close in 3 seconds...");
            Console.ResetColor();
            Thread.Sleep(3000);
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Launch failed: " + ex.Message);
            Console.WriteLine("\nPress any key to exit...");
            Console.ResetColor();
            Console.ReadKey();
        }
    }
}
