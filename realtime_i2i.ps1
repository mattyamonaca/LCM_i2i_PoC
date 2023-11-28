Set-Location $PSScriptRoot

.\venv\Scripts\activate

python "realtime_i2i.py"

Read-Host | Out-Null ;