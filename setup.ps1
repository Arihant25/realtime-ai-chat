# One command: install everything, then open the web UI.
#   powershell -ExecutionPolicy Bypass -File setup.ps1 [--qwen] [--no-orpheus]
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "== Installing uv"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "== Python dependencies"
uv sync
if ($LASTEXITCODE) { exit $LASTEXITCODE }

uv run realtime-chat-setup @args
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "`n== Starting the web UI on http://127.0.0.1:8890 (Ctrl+C to stop)"
uv run realtime-chat-web
