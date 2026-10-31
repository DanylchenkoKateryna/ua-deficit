# Renders every page of the finished Word file to PNG with Microsoft Word, for visual checking.
param(
    [string]$Doc = "C:\Projects\ua-deficit-forecast\docs\practical_part.docx",
    [string]$OutDir = "C:\Projects\ua-deficit-forecast\docs\qa"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png | Remove-Item -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $true
$word.DisplayAlerts = 0
try {
    $d = $word.Documents.Open($Doc, $false, $true)
    $word.ActiveWindow.View.Type = 3   # print layout
    $n = $d.ComputeStatistics(2)
    for ($i = 1; $i -le $n; $i++) {
        $page = $word.ActiveWindow.ActivePane.Pages.Item($i)
        $bytes = $page.EnhMetaFileBits
        $emf = Join-Path $OutDir ("page-{0:D2}.emf" -f $i)
        [System.IO.File]::WriteAllBytes($emf, $bytes)
        $img = [System.Drawing.Image]::FromFile($emf)
        $bmp = New-Object System.Drawing.Bitmap 1000, 1414
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.Clear([System.Drawing.Color]::White)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.DrawImage($img, 0, 0, 1000, 1414)
        $g.Dispose(); $img.Dispose()
        $bmp.Save((Join-Path $OutDir ("page-{0:D2}.png" -f $i)), [System.Drawing.Imaging.ImageFormat]::Png)
        $bmp.Dispose()
        Remove-Item $emf -Force
    }
    $d.Close($false)
    Write-Output "rendered $n pages"
} finally {
    $word.Quit()
}
