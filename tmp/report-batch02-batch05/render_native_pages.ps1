$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$reportWork = Split-Path -Parent $PSCommandPath
$reportRoot = Split-Path -Parent (Split-Path -Parent $reportWork)
$reportSource = Join-Path $reportRoot 'models/reports/Laporan_Eksperimen_Batch_02_sampai_05.docx'
$reportImages = Join-Path $reportWork 'native_pages'
[void](New-Item -ItemType Directory -Path $reportImages -Force)
$reportOwnInstance = $false
$reportApp = $null
if (-not (Get-Process WINWORD -ErrorAction SilentlyContinue)) {
    [void](Start-Process -FilePath 'C:/Program Files/Microsoft Office/Root/Office16/WINWORD.EXE' -ArgumentList @('/a', '/q', '/n', ('"' + $reportSource + '"')) -WindowStyle Hidden -PassThru)
    $reportOwnInstance = $true
}
for ($reportTry = 0; $reportTry -lt 40; $reportTry++) {
    try {
        $reportApp = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
        if ($reportApp.Documents.Count -eq 1 -and $reportApp.ActiveDocument.FullName -eq $reportSource) { break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
if ($null -eq $reportApp) { throw 'Report Word instance unavailable.' }
$reportDoc = $reportApp.ActiveDocument
if ($reportDoc.FullName -ne $reportSource) { throw 'Active document is not the requested report.' }
$reportPages = $reportDoc.ActiveWindow.Panes.Item(1).Pages
$reportCount = $reportPages.Count
$reportMetadata = @()
try {
    for ($reportIndex = 1; $reportIndex -le $reportCount; $reportIndex++) {
        [byte[]]$reportBytes = $reportPages.Item($reportIndex).EnhMetaFileBits
        $reportStream = [System.IO.MemoryStream]::new($reportBytes, $false)
        $reportImage = [System.Drawing.Image]::FromStream($reportStream)
        $reportBitmap = [System.Drawing.Bitmap]::new(1275, 1650)
        $reportGraphics = [System.Drawing.Graphics]::FromImage($reportBitmap)
        $reportGraphics.Clear([System.Drawing.Color]::White)
        $reportGraphics.DrawImage($reportImage, 0, 0, 1275, 1650)
        $reportOutput = Join-Path $reportImages ('page-' + $reportIndex + '.png')
        $reportBitmap.Save($reportOutput, [System.Drawing.Imaging.ImageFormat]::Png)
        $reportGraphics.Dispose()
        $reportBitmap.Dispose()
        $reportImage.Dispose()
        $reportStream.Dispose()
        $reportStart = $reportDoc.GoTo(1, 1, $reportIndex).Start
        $reportEnd = if ($reportIndex -lt $reportCount) { $reportDoc.GoTo(1, 1, ($reportIndex + 1)).Start } else { $reportDoc.Content.End }
        $reportText = $reportDoc.Range($reportStart, $reportEnd).Text
        $reportMetadata += @{page=$reportIndex; text=$reportText}
        Write-Output ('PAGE ' + $reportIndex + ' chars=' + $reportText.Length + ' ' + $reportOutput)
    }
    $reportMetadata | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $reportWork 'native_page_text.json') -Encoding UTF8
} finally {
    if ($reportOwnInstance) {
        $reportDoc.Close(0)
        $reportApp.Quit()
    }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportDoc)
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($reportApp)
}
