param(
  [string]$SrcDir = "results\metrics",
  [string]$DstDir = "docs\figures",
  [switch]$NoReadme
)

$ErrorActionPreference = "Stop"

# Ensure destination exists
New-Item -ItemType Directory -Force -Path $DstDir | Out-Null

# Pick latest compare plots
$heat = Get-ChildItem $SrcDir -Filter "robust_compare_delta_heatmap.png" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$pgd  = Get-ChildItem $SrcDir -Filter "robust_compare_pgd10.png"       | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $heat -or -not $pgd) {
  Write-Host "[warn] Could not find expected compare plots in $SrcDir" -ForegroundColor Yellow
  exit 1
}

Copy-Item $heat.FullName (Join-Path $DstDir "xai_robust_debug_delta_heatmap.png") -Force
Copy-Item $pgd.FullName  (Join-Path $DstDir "xai_robust_debug_pgd10.png")        -Force
Write-Host "[ok] Exported thesis figures to $DstDir" -ForegroundColor Green

if (-not $NoReadme) {
  $readme = Get-Content .\README.md -Raw
  $snippetLines = @(
    '### Thesis figures (debug)',
    '',
    '- Delta heatmap: `docs/figures/xai_robust_debug_delta_heatmap.png`',
    '- PGD10 comparison: `docs/figures/xai_robust_debug_pgd10.png`'
  )
  $snippet = [string]::Join("`r`n", $snippetLines)
  if ($readme -notmatch "### Thesis figures \(debug\)") {
    $readme = $readme.TrimEnd() + "`r`n`r`n" + $snippet + "`r`n"
    Set-Content .\README.md -Value $readme -Encoding utf8
    Write-Host "[ok] Updated README with Thesis figures section" -ForegroundColor Green
  } else {
    Write-Host "[skip] Thesis figures section already present" -ForegroundColor Yellow
  }
}
