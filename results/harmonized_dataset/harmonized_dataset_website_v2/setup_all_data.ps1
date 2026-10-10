$ErrorActionPreference = "Stop"

# This script is intended to live in:
# repo\results\harmonized_dataset\final_dataset_website\
$site = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $site "..\..\..")).Path
$results = Join-Path $repo "results\harmonized_dataset"
$destination = Join-Path $site "data\project_outputs"

Write-Host "Repository: $repo"
Write-Host "Source results: $results"
Write-Host "Website data: $destination"

if (!(Test-Path $results)) {
    throw "Could not find results\harmonized_dataset at $results"
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null

# Copy every generated harmonized-dataset result folder except the website itself.
Get-ChildItem -Path $results -Force | ForEach-Object {
    if ($_.Name -in @("final_dataset_website", "harmonized_dataset_website_v2")) {
        return
    }

    $target = Join-Path $destination $_.Name

    if ($_.PSIsContainer) {
        Write-Host "Copying folder: $($_.Name)"
        Copy-Item -Path $_.FullName -Destination $target -Recurse -Force
    }
    else {
        Write-Host "Copying file: $($_.Name)"
        Copy-Item -Path $_.FullName -Destination $target -Force
    }
}

Write-Host ""
Write-Host "ALL GENERATED DATASET OUTPUTS HAVE BEEN COPIED INTO THE WEBSITE." -ForegroundColor Green
Write-Host ""
Write-Host "Now run:" -ForegroundColor Cyan
Write-Host "  python -m http.server 8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then open:" -ForegroundColor Cyan
Write-Host "  http://localhost:8000" -ForegroundColor Yellow
