# Input figure QA

Six figures were rendered in Python and inspected individually: A1/A120 mean intensity, A1/A120 750nm intensity, native spectra and response distribution. Axes, labels and orientation are legible and consistent with the actual cube geometry; no axis permutation or selective image manipulation was applied. All240 records contribute to the histogram; fixed two-ID lines and images were declared before any formal OOF prediction.

Each figure has one plot area, so comparable multi-panel alignment is not applicable. PDFs passed the rendered text audit at a minimum8pt (required5pt) and the collision audit with zero FAIL and zero WARN. The native-spectrum collision report contains two intentional fully contained legend-fill overlaps, with readable text inside the legend box; visual review confirms no interference with curves.

Static source preflight has no FAIL. Its two WARN concern PNG preview rather than submission TIFF and300dpi rather than the default600dpi; these are intentionally internal QC previews alongside editable SVG/PDF, without a journal submission claim. The final width is88.9mm. Quantitative model figures remain ungenerated pending complete cluster OOF, and therefore are not covered by this QA.
