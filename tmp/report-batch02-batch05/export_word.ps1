$ErrorActionPreference = 'Stop'
$reportWork = Split-Path -Parent $PSCommandPath
$reportRoot = Split-Path -Parent (Split-Path -Parent $reportWork)
$reportPath = Join-Path $reportRoot 'models/reports/Laporan_Eksperimen_Batch_02_sampai_05.docx'
$reportPdf = Join-Path $reportWork 'word_render.pdf'
$reportApp = $null
$reportDoc = $null
$reportProcess = $null
if (Get-Process WINWORD -ErrorAction SilentlyContinue) { throw 'Word is already running; no existing instance will be used.' }
try {
    $reportProcess = Start-Process -FilePath 'C:/Program Files/Microsoft Office/Root/Office16/WINWORD.EXE' -ArgumentList @('/a', '/q', '/n', ('"' + $reportPath + '"')) -WindowStyle Hidden -PassThru
    Write-Output ('Started isolated Word process ' + $reportProcess.Id)
    for ($reportTry = 0; $reportTry -lt 40; $reportTry++) {
        try { $reportApp = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application'); break } catch { Start-Sleep -Milliseconds 500 }
    }
    if ($null -eq $reportApp) { throw 'Word automation did not become available.' }
    Write-Output 'Connected to isolated Word process'
    $reportApp.Visible = $false
    $reportApp.DisplayAlerts = 0
    $reportApp.AutomationSecurity = 3
    $reportDoc = $reportApp.ActiveDocument
    if ($reportDoc.FullName -ne $reportPath) { throw 'Active document does not match report; no export performed.' }
    Write-Output 'Verified report document'
    $reportDoc.Repaginate()
    $reportDoc.ExportAsFixedFormat($reportPdf, 17)
    Write-Output $reportPdf
} finally {
    if ($null -ne $reportDoc) {
        $reportDoc.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDoc)
    }
    if ($null -ne $reportApp) {
        $reportApp.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportApp)
    } elseif ($null -ne $reportProcess -and -not $reportProcess.HasExited) {
        Stop-Process -Id $reportProcess.Id -Force
    }
}
