python -c "import sys; print(sys.executable)"
python -V
pip freeze | Out-File -Encoding utf8 requirements.freeze.txt
Write-Host "Wrote requirements.freeze.txt"
