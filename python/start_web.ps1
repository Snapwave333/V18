$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python\.venv\Scripts\python.exe"
$scriptPath = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python\web_server.py"
$workDir = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python"

$proc = Start-Process -FilePath $pythonExe -ArgumentList $scriptPath -WorkingDirectory $workDir -PassThru -WindowStyle Normal
Write-Host "Started web server with PID: $($proc.Id)"
Start-Sleep 2

if (!$proc.HasExited) {
    Write-Host "Web server is running"
} else {
    Write-Host "Web server exited with code: $($proc.ExitCode)"
}
