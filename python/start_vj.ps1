$ErrorActionPreference = "Stop"
$pythonExe = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python\.venv\Scripts\python.exe"
$scriptPath = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python\main.py"
$workDir = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine\python"

$proc = Start-Process -FilePath $pythonExe -ArgumentList $scriptPath -WorkingDirectory $workDir -PassThru -WindowStyle Normal
Write-Host "Started VJ Engine with PID: $($proc.Id)"
