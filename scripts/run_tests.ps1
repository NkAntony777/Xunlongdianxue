# 分范围测试入口（PowerShell）
# 用法:
#   .\scripts\run_tests.ps1
#   .\scripts\run_tests.ps1 frontend
#   .\scripts\run_tests.ps1 engine-fast
#   .\scripts\run_tests.ps1 auto
#   .\scripts\run_tests.ps1 all
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = Join-Path $Root "engine\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "找不到 venv Python: $Py"
}
$env:PYTHONIOENCODING = "utf-8"
& $Py -X utf8 (Join-Path $Root "scripts\run_tests.py") @args
exit $LASTEXITCODE
