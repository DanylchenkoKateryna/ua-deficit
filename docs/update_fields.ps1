# Opens a generated Word file in Microsoft Word, updates every field (table of contents, page numbers) and saves it
# as the study guide. If the target file is locked (open in Word), saves next to it with the suffix _v3.
# The script is ASCII only (Windows PowerShell 5.1 reads BOM-less files as ANSI), so the Cyrillic name is built from escapes.
param(
    [string]$In = "C:\Projects\ua-deficit-forecast\docs\guide_raw.docx",
    [string]$Out = ""
)
$ErrorActionPreference = "Stop"
$name = [regex]::Unescape("\u041f\u0456\u0434\u0433\u043e\u0442\u043e\u0432\u043a\u0430_\u0434\u043e_\u0437\u0430\u0445\u0438\u0441\u0442\u0443")
if ($Out -eq "") { $Out = Join-Path "C:\Projects\ua-deficit-forecast\docs" ($name + ".docx") }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($In)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
    Write-Output ("pages=" + $doc.ComputeStatistics(2))
    try {
        $doc.SaveAs2($Out, 16)
        Write-Output ("saved " + $Out)
    } catch {
        $alt = [System.IO.Path]::ChangeExtension($Out, $null).TrimEnd('.') + "_v3.docx"
        Write-Output ("target locked, saving " + $alt)
        $doc.SaveAs2($alt, 16)
    }
    $doc.Close($false)
} finally {
    $word.Quit()
}
