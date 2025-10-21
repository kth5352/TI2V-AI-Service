$env:HF_HUB_ENABLE_HF_TRANSFER=0

param([switch]$ForceBuild)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "=== WAN I2V run.ps1 ===" -ForegroundColor Cyan

# Python venv
if (!(Test-Path "$root\.venv")) {
  py -3.11 -m venv .venv
}
. "$root\.venv\Scripts\Activate.ps1"

# Backend deps
Set-Location "$root\backend"
pip install --upgrade pip
pip install -r requirements.txt

# 모델 받기 (이미 있으면 스킵)
if (!(Test-Path "$root\backend\models\Wan2.2-TI2V-5B-Diffusers")) {
  pip install "huggingface_hub[cli]"
  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$root\backend\models\Wan2.2-TI2V-5B-Diffusers"
}

# downloads 폴더
if (!(Test-Path "$root\backend\downloads")) { New-Item -ItemType Directory "$root\backend\downloads" | Out-Null }

# Frontend deps
Set-Location "$root\frontend"
if ($ForceBuild -or -not (Test-Path "$root\frontend\node_modules")) {
  npm ci; if (!$?) { npm install }
}

# 백엔드와 프론트 동시 실행
Set-Location "$root"

# --- 교체 후 (안정 버전) ---
# 1) 부모 셸에서 환경변수 설정
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

# 2) 백엔드/프론트 각각 실행
Start-Process powershell -ArgumentList "-NoExit","-Command","cd backend; uvicorn main:app --host 0.0.0.0 --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","cd frontend; npm run dev"