$ErrorActionPreference = "Stop"

function Assert-Contains {
    param(
        [string]$Content,
        [string]$Pattern,
        [string]$Message
    )

    if ($Content -notmatch $Pattern) {
        throw $Message
    }
}

$root = Split-Path -Parent $PSScriptRoot
$requiredFiles = @(
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "artifacts/life_of_a_bill.joblib"
)

foreach ($relativePath in $requiredFiles) {
    $path = Join-Path $root $relativePath
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required deployment file: $relativePath"
    }
}

$readme = Get-Content -Raw (Join-Path $root "README.md")
$dockerfile = Get-Content -Raw (Join-Path $root "Dockerfile")
$requirements = Get-Content -Raw (Join-Path $root "requirements.txt")
$app = Get-Content -Raw (Join-Path $root "app.py")

Assert-Contains $readme "title:\s*The Life of a Bill" "README must define the Hugging Face Space title."
Assert-Contains $readme "sdk:\s*docker" "README must configure the Docker SDK."
Assert-Contains $readme "app_port:\s*8501" "README must expose Streamlit port 8501."
$shortDescriptionMatch = [regex]::Match($readme, "(?m)^short_description:\s*(.+)$")
if (-not $shortDescriptionMatch.Success) {
    throw "README must define a Hugging Face short_description."
}
if ($shortDescriptionMatch.Groups[1].Value.Trim().Length -gt 60) {
    throw "Hugging Face short_description must be 60 characters or fewer."
}
Assert-Contains $dockerfile "EXPOSE\s+8501" "Dockerfile must expose port 8501."
Assert-Contains $dockerfile "streamlit.*app\.py" "Dockerfile must run the Streamlit entry point."
Assert-Contains $requirements "streamlit" "requirements.txt must include Streamlit."
Assert-Contains $requirements "xgboost" "requirements.txt must include XGBoost."
Assert-Contains $app "Try a sample scenario" "App must provide sample scenarios."
Assert-Contains $app "Historical pattern, not a forecast" "App must show the evaluation disclaimer."
Assert-Contains $app "First Reading" "App must display the First Reading class."
Assert-Contains $app "Second Reading" "App must display the Second Reading class."
Assert-Contains $app "Approved" "App must display the Approved class."
Assert-Contains $app "predict_bill" "App must call the reusable prediction core."

Write-Output "Deployment verification passed."
