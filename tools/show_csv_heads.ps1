Param(
  [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
  [string[]]$Paths,
  [int]$Lines = 8
)

foreach ($p in $Paths) {
  if (-not (Test-Path $p)) { Write-Warning "Not found: $p"; continue }
  Write-Host "`n== $p" -ForegroundColor Cyan
  try {
    # Pretty table for CSVs with headers
    $table = Import-Csv $p | Select-Object -First $Lines | Format-Table -AutoSize | Out-String -Width 4096
    if (-not [string]::IsNullOrWhiteSpace($table)) { Write-Host $table.TrimEnd() }
    else { throw "Empty or unparseable CSV" }
  }
  catch {
    # Fallback: raw first lines (header + rows)
    Write-Host "(fallback: raw head)"
    Get-Content $p -First ($Lines + 1) | ForEach-Object { Write-Host $_ }
  }
}
