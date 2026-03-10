$ErrorActionPreference = "Stop"

# 1. Kill stale processes
Write-Host "Cleaning up stale processes..." -ForegroundColor Cyan
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process node* -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process electron* -ErrorAction SilentlyContinue | Stop-Process -Force

$vjDir = "C:\Users\chrom\Desktop\Vibes\v18\ollama-vj-engine"
$pythonExe = "$vjDir\python\.venv\Scripts\python.exe"

# 2. Start VJ Engine (Pygame App)
Write-Host "Launching VIBES V18..." -ForegroundColor Green
Set-Location -Path "$vjDir\python"
& $pythonExe main.py --worker deform --delay 0

Write-Host "`nSYSTEM SHUTDOWN." -ForegroundColor Magenta
