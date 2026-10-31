# Opens the generated Word file in Microsoft Word, fixes the table-of-contents styles, updates all fields
# (table of contents, page numbers), saves the result and exports a PDF for checking.
param(
    [string]$In = "C:\Projects\ua-deficit-forecast\docs\practical_part_raw.docx",
    [string]$Out = "C:\Projects\ua-deficit-forecast\docs\practical_part.docx",
    [string]$Pdf = "C:\Projects\ua-deficit-forecast\docs\practical_part_check.pdf"
)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($In)
    # TOC styles: chapters bold, sections indented, Times New Roman 14, 1.5 spacing, dot leader to the page number
    foreach ($pair in @(@(-20, $true, 0.0), @(-21, $false, 1.25))) {
        $st = $doc.Styles.Item($pair[0])
        $st.Font.Name = "Times New Roman"; $st.Font.Size = 14; $st.Font.Bold = [int]$pair[1]
        $st.ParagraphFormat.LeftIndent = $word.CentimetersToPoints($pair[2])
        $st.ParagraphFormat.FirstLineIndent = 0
        $st.ParagraphFormat.SpaceBefore = 0; $st.ParagraphFormat.SpaceAfter = 0
        $st.ParagraphFormat.LineSpacingRule = 5   # multiple
        $st.ParagraphFormat.LineSpacing = $word.LinesToPoints(1.5)
        $st.ParagraphFormat.Alignment = 0
    }
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)
    $doc.SaveAs2($Out, 16)
    $doc.ExportAsFixedFormat($Pdf, 17)
    Write-Output "pages=$pages"
    $doc.Close($false)
} finally {
    $word.Quit()
}
