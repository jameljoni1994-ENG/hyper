# Supervisor: alternate E6 / E8 sessions (STRICTLY sequential -- wall-clock
# timing experiments must never overlap) until both report todo=0, then run
# analysis.py. Lives in-project because %TEMP% gets wiped externally.
# Logs per session under results\logs\.
$ErrorActionPreference = "Continue"
$py   = "C:\Users\Windows.11\AppData\Local\Programs\Python\Python313\python.exe"
$root = "C:\Users\Windows.11\Desktop\hyper"
$code = Join-Path $root "code"
$logd = Join-Path $root "results\logs"
New-Item -ItemType Directory -Force -Path $logd | Out-Null
$trace = Join-Path $logd "supervisor_trace.log"

function Trace($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $trace -Value $line
}

Trace "supervisor started (pid $PID)"

$e6_done = $false
$e8_done = $false
$round = 0
$deadline = (Get-Date).AddHours(14)

while (-not ($e6_done -and $e8_done)) {
    if ((Get-Date) -gt $deadline) {
        Trace "DEADLINE hit after $round rounds; exiting without analysis"
        exit 1
    }
    $round++
    foreach ($exp in @("E6", "E8")) {
        $doneVar = Get-Variable ("{0}_done" -f $exp) -ValueOnly
        if ($doneVar) { continue }
        if ($exp -eq "E6") { $budget = 60 } else { $budget = 25 }
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $log = Join-Path $logd ("session_{0}_{1}.log" -f $exp, $stamp)
        Trace "launch round=$round exp=$exp budget=${budget}min log=$log"
        & $py (Join-Path $code "run_all_v2.py") --exp $exp `
            --budget-min $budget 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            Trace "WARNING: exp=$exp exited with code $LASTEXITCODE"
        }
        $head = Select-String -Path $log -Pattern "todo=(\d+)" |
            Select-Object -First 1
        if ($head -and $head.Matches[0].Groups[1].Value -eq "0") {
            Trace "exp=$exp reports todo=0 -> COMPLETE"
            Set-Variable -Name ("{0}_done" -f $exp) -Value $true
        } else {
            Trace "exp=$exp session finished; more work remains"
        }
    }
}

Trace "all experiments complete; running analysis.py"
& $py (Join-Path $code "analysis.py") 2>&1 |
    Tee-Object -FilePath (Join-Path $logd "analysis_final.log")
Trace "analysis exit code: $LASTEXITCODE"
Trace "supervisor done"
