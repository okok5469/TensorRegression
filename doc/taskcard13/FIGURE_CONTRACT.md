# Apple figures: evidence and QA contract

Data QC asks whether the actual two spatial axes, zero-coded pixels and native spectral axis have been preserved. Fixed IDs A1 and A120 were declared before any formal OOF result. Each image uses the original complete rectangle, unchanged orientation and a global viridis display range chosen automatically from that image. No segmentation, rotation, image stitching or selective local adjustment is applied. The scale is relative ROI position; physical pixel size is unknown. Recorded intensity is not local dry matter.

The separate native spectra and response histogram describe the input population. Individual curves are raw per-ID spatial means, with no across-entity uncertainty claim. The histogram uses all 240 records. Model figures will only be generated after complete OOF coverage; repeat1 is fixed in advance for all scatter plots.

Backend: Python/matplotlib, following this project's Python workflow. Every current QC figure has one plot area, so comparable multi-panel alignment is not applicable. Editable SVG/PDF and PNG previews are exported with 8-point type at88.9mm width. Run source, PDF glyph-size and rendered collision audits; inspect all panels. Source data are the immutable MAT and audited data.npz, and exclusion count is zero. These are internal empirical-report figures without a journal submission claim.
