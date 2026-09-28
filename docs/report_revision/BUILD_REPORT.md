# Rebuilding the report

I export the report from its Markdown source and evaluation figures. For Figure 4.2, I select the manual camera screenshot at `docs/report_revision/assets/working_camera_dashboard.png` and its current caption. The builder defaults to the historical screenshot, so I select the current figure during this export.

I run this from the project root in my Python environment:

```powershell
@'
from pathlib import Path
import importlib.util
import sys

root = Path.cwd()
spec = importlib.util.spec_from_file_location("report_builder", root / "tools/build_week18_report_pdf.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
original_figure = builder.figure_for

def report_figure(marker, figure_dir=builder.FIGURE_DIR):
    if marker.startswith("Figure 4.2"):
        return (
            builder.Image(str(root / "docs/report_revision/assets/working_camera_dashboard.png"), width=160 * builder.mm, height=90 * builder.mm),
            "Figure 4.2 - Manual CabInspector demonstration with a usable camera image; display and status evidence, not accuracy."
        )
    return original_figure(marker, figure_dir)

builder.figure_for = report_figure
sys.argv = ["build_report", "--report", str(root / "docs/report_revision/CabInspector_Final_Report_Revised.md"), "--figure-dir", str(root / "docs/report_revision/figures"), "--output", str(root / "output/pdf/CabInspector_Final_Report_Revised_Orange.pdf")]
builder.main()
'@ | python -
```

I check the chapter limits with `python tools/verify_report_limits.py --report docs/report_revision/CabInspector_Final_Report_Revised.md` and inspect the exported pages before using the PDF. The screenshot demonstrates a usable camera image and visible status output; I still have no representative visual accuracy benchmark.
