# Renders every slide of the presentation to PNG with Microsoft PowerPoint, for visual checking.
# Uses the newest .pptx in the docs folder (the _v3 copy when the main file was locked), or the file given with -File.
param(
    [string]$Dir = "C:\Projects\ua-deficit-forecast\docs",
    [string]$OutDir = "C:\Projects\ua-deficit-forecast\docs\qa_slides",
    [string]$File = ""
)
$ErrorActionPreference = "Stop"
if ($File -eq "") {
    $File = (Get-ChildItem $Dir -Filter *.pptx | Where-Object { $_.Name -notlike '~$*' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png | Remove-Item -Force
$ppt = New-Object -ComObject PowerPoint.Application
try {
    $pres = $ppt.Presentations.Open($File, -1, 0, 0)   # read-only, no title, no window
    $pres.Export($OutDir, "PNG", 1280, 720)
    Write-Output ("slides=" + $pres.Slides.Count)
    $pres.Close()
} finally {
    $ppt.Quit()
}
Get-ChildItem $OutDir -Filter *.PNG | Select-Object -First 3 Name
