function Invoke-Py {
  param([string]$Code)
  $tmp = [System.IO.Path]::GetTempFileName() + '.py'
  Set-Content -LiteralPath $tmp -Value $Code -Encoding UTF8
  python $tmp
  Remove-Item $tmp -Force
}

function Invoke-PyModule {
  param(
    [Parameter(Mandatory)] [string] $Module,
    [Parameter(Mandatory)] [hashtable] $Args
  )
  $argList = @($Module)
  foreach ($k in $Args.Keys) {
    $val = $Args[$k]
    if ($null -ne $val -and $val -ne $false) {
      $argList += "--$k"
      if ($val -isnot [bool]) { $argList += "$val" }
    }
  }
  python -m @argList
}

function Use-RepoPYTHONPATH { $env:PYTHONPATH = "$PWD" }
