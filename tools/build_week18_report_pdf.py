"""Build the CabInspector final report from Markdown and evaluation figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "report_revision" / "CabInspector_Final_Report_Revised.md"
OUTPUT = ROOT / "output" / "pdf" / "CabInspector_Final_Report.pdf"
FIGURE_DIR = ROOT / "docs" / "report_revision" / "figures"
COMBINED_ACCEPTANCE_SCREENSHOT = ROOT / "docs" / "report_revision" / "assets" / "dashboard_snapshot.png"

FONT_REGULAR = r"C:\Windows\Fonts\ARIALUNI.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
ARABIC_RE = re.compile(r"[\u0600-\u06FF]+")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("CabinRegular", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("CabinBold", FONT_BOLD))


def clean_text(value: str) -> str:
    """Shape Arabic runs and normalise punctuation for predictable PDF output."""

    replacements = {
        "—": " - ",
        "–": "-",
        "“": '"',
        "”": '"',
        "’": "'",
        "×": "x",
        "…": "...",
        "‑": "-",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)

    def shape(match: re.Match[str]) -> str:
        return get_display(arabic_reshaper.reshape(match.group(0)))

    return ARABIC_RE.sub(shape, value)


def inline_markup(value: str) -> str:
    value = clean_text(value)
    value = escape(value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


class DiagramBase(Flowable):
    def __init__(self, *, diagram_height=76 * mm):
        super().__init__()
        # Flowable.__init__ starts with zero dimensions, so set explicit
        # instance dimensions. Otherwise Platypus reserves no page space and
        # the diagram is drawn over surrounding report text.
        self.width = 170 * mm
        self.height = diagram_height

    def wrap(self, available_width, available_height):
        return min(self.width, available_width), self.height

    def box(self, canvas, x, y, width, height, text, *, fill, stroke=HexColor("#52616B"), text_color=HexColor("#172B4D")):
        canvas.setStrokeColor(stroke)
        canvas.setFillColor(fill)
        canvas.roundRect(x, y, width, height, 5, stroke=1, fill=1)
        canvas.setFillColor(text_color)
        canvas.setFont("CabinBold", 7.2)
        lines = text.split("\n")
        start_y = y + height / 2 + (len(lines) - 1) * 4.3
        for index, line in enumerate(lines):
            canvas.drawCentredString(x + width / 2, start_y - index * 8.6, line)

    def arrow(self, canvas, x1, y1, x2, y2, *, colour=HexColor("#52616B"), dashed=False):
        canvas.setStrokeColor(colour)
        canvas.setLineWidth(1.15)
        canvas.setDash(3, 2) if dashed else canvas.setDash()
        canvas.line(x1, y1, x2, y2)
        canvas.setDash()
        angle_x, angle_y = x2 - x1, y2 - y1
        length = max((angle_x**2 + angle_y**2) ** 0.5, 1)
        ux, uy = angle_x / length, angle_y / length
        left_x, left_y = x2 - 7 * ux + 3 * uy, y2 - 7 * uy - 3 * ux
        right_x, right_y = x2 - 7 * ux - 3 * uy, y2 - 7 * uy + 3 * ux
        canvas.line(x2, y2, left_x, left_y)
        canvas.line(x2, y2, right_x, right_y)


class ArchitectureDiagram(DiagramBase):
    def draw(self):
        c = self.canv
        c.setFillColor(HexColor("#FAFBFC"))
        c.roundRect(0, 0, self.width, self.height - 2, 8, stroke=0, fill=1)
        inputs = [
            (8, 160, "Live webcam", HexColor("#E9F2FF")),
            (185, 160, "Live microphone", HexColor("#E9F2FF")),
            (362, 160, "Public recorded\ntelemetry trip", HexColor("#E9F2FF")),
        ]
        models = [
            (8, 112, "Visual pipeline\nMediaPipe / YOLOv8 / Haar", HexColor("#EDF9F0")),
            (185, 112, "Audio pipeline\nYAMNet / Whisper / loudness", HexColor("#EDF9F0")),
            (362, 112, "Telemetry replay\nPRIMUS + linear head", HexColor("#EDF9F0")),
        ]
        for x, y, label, fill in inputs:
            self.box(c, x, y, 145, 27, label, fill=fill)
        for x, y, label, fill in models:
            self.box(c, x, y, 145, 35, label, fill=fill)
        for x in (80, 257, 434):
            self.arrow(c, x, 160, x, 147)
            self.arrow(c, x, 112, 257, 82)
        self.box(c, 172, 58, 170, 25, "Shared evidence state", fill=HexColor("#FFF4D9"), stroke=HexColor("#F2994A"))
        self.box(c, 172, 23, 170, 25, "Transparent rules and bounded risk", fill=HexColor("#FFF4D9"), stroke=HexColor("#F2994A"))
        self.arrow(c, 257, 58, 257, 48)
        self.box(c, 2, 2, 145, 15, "Dashboard: audio / camera / risk", fill=HexColor("#F5EEFF"), stroke=HexColor("#9B51E0"))
        self.box(c, 184, 2, 145, 15, "Structured event log", fill=HexColor("#F5EEFF"), stroke=HexColor("#9B51E0"))
        self.box(c, 367, 2, 145, 15, "Optional screenshots", fill=HexColor("#F5EEFF"), stroke=HexColor("#9B51E0"))
        for x in (74, 256, 440):
            self.arrow(c, 257, 23, x, 17)


class DecisionFlowDiagram(DiagramBase):
    def draw(self):
        c = self.canv
        c.setFillColor(HexColor("#FAFBFC"))
        c.roundRect(0, 0, self.width, self.height - 2, 8, stroke=0, fill=1)
        labels = [
            "Model or\nsensor output",
            "Context and\nvalidity checks",
            "Persistence /\nconfidence smoothing",
            "Source-labelled\nevent state",
            "Bounded risk\ncontribution",
        ]
        x_positions = [5, 105, 205, 305, 405]
        for x, label in zip(x_positions, labels):
            self.box(c, x, 135, 88, 36, label, fill=HexColor("#EAF3FF"), stroke=HexColor("#2F80ED"))
        for x in x_positions[:-1]:
            self.arrow(c, x + 88, 153, x + 100, 153)
        self.box(c, 40, 65, 155, 34, "Audio speech: in-memory utterance gate\nRaw audio discarded by default", fill=HexColor("#EDF9F0"), stroke=HexColor("#27AE60"))
        self.box(c, 230, 65, 155, 34, "Opt-in only: transcript store\nand readable export", fill=HexColor("#EDF9F0"), stroke=HexColor("#27AE60"))
        self.box(c, 400, 65, 90, 34, "Dashboard alert\nand explanation", fill=HexColor("#FFF4D9"), stroke=HexColor("#F2994A"))
        self.box(c, 400, 16, 90, 34, "Structured event log\n(no transcript text)", fill=HexColor("#FFF4D9"), stroke=HexColor("#F2994A"))
        self.arrow(c, 49, 135, 117, 99, dashed=True)
        self.arrow(c, 195, 82, 230, 82, dashed=True)
        self.arrow(c, 449, 135, 445, 99)
        self.arrow(c, 445, 65, 445, 50)


class DashboardDiagram(DiagramBase):
    def __init__(self):
        super().__init__(diagram_height=68 * mm)

    def draw(self):
        c = self.canv
        total_height = self.height - 2
        c.setFillColor(HexColor("#251D19"))
        c.roundRect(0, 0, self.width, total_height, 8, stroke=0, fill=1)
        self.box(c, 12, 16, 125, total_height - 32, "Cabin Audio\n\nInput / language\nVoice level\nSafety state\nTranscript controls", fill=HexColor("#312722"), stroke=HexColor("#5E514A"), text_color=colors.white)
        self.box(c, 157, 16, 168, total_height - 32, "Live camera view\n\nLandmarks and safe zone\nObject evidence\nNon-blocking visual loop", fill=HexColor("#3A302A"), stroke=HexColor("#8B6B5A"), text_color=colors.white)
        self.box(c, 345, 16, 125, total_height - 32, "Driver Monitor\n\nCamera status\nTelemetry replay\nRisk breakdown\nAlerts and controls", fill=HexColor("#312722"), stroke=HexColor("#5E514A"), text_color=colors.white)
        c.setFillColor(HexColor("#00B84A"))
        c.roundRect(222, total_height - 18, 40, 12, 2, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("CabinBold", 6.8)
        c.drawCentredString(242, total_height - 14, "LIVE")


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(HexColor("#D9E2EC"))
    canvas.line(doc.leftMargin, 13 * mm, A4[0] - doc.rightMargin, 13 * mm)
    canvas.setFillColor(HexColor("#5E6C84"))
    canvas.setFont("CabinRegular", 8)
    canvas.drawRightString(A4[0] - doc.rightMargin, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="CabinBold", fontSize=26, leading=31, textColor=colors.black, alignment=TA_CENTER, spaceAfter=14),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="CabinRegular", fontSize=13, leading=18, textColor=colors.black, alignment=TA_CENTER),
        "chapter": ParagraphStyle("chapter", parent=base["Heading1"], fontName="CabinBold", fontSize=19, leading=24, textColor=colors.black, spaceBefore=0, spaceAfter=12),
        "section": ParagraphStyle("section", parent=base["Heading2"], fontName="CabinBold", fontSize=13, leading=17, textColor=colors.black, spaceBefore=12, spaceAfter=6, keepWithNext=1),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="CabinRegular", fontSize=10.3, leading=14.4, spaceAfter=7, alignment=TA_LEFT, allowWidows=0, allowOrphans=0),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="CabinRegular", fontSize=10.1, leading=13.5, leftIndent=14, firstLineIndent=0, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="CabinRegular", fontSize=8.5, leading=11, textColor=HexColor("#52616B"), alignment=TA_CENTER, spaceBefore=3, spaceAfter=12),
        "reference": ParagraphStyle("reference", parent=base["BodyText"], fontName="CabinRegular", fontSize=8.7, leading=11.5, leftIndent=10, firstLineIndent=-10, spaceAfter=4),
        "code": ParagraphStyle("code", parent=base["Code"], fontName="Courier", fontSize=8, leading=10, backColor=HexColor("#F4F5F7"), borderColor=HexColor("#D9E2EC"), borderWidth=0.5, borderPadding=7, spaceBefore=4, spaceAfter=8),
    }


def make_table(lines, style_map):
    rows = []
    header_style = ParagraphStyle(
        "table_header",
        parent=style_map["body"],
        fontName="CabinBold",
        textColor=colors.black,
    )
    for row_number, line in enumerate(lines):
        values = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if row_number == 0 and values and values[0] == "Domain / component":
            model_table_widths = [31 * mm, 53 * mm, 37 * mm, 55 * mm]
        if not values or all(re.fullmatch(r"[:-]+", value.replace(" ", "")) for value in values):
            continue
        cell_style = header_style if not rows else style_map["body"]
        rows.append([Paragraph(inline_markup(value), cell_style) for value in values])
    column_count = max((len(row) for row in rows), default=1)
    if column_count == 4 and rows[0][0].getPlainText() == "Domain / component":
        widths = model_table_widths
    elif column_count == 4:
        widths = [38 * mm, 28 * mm, 45 * mm, 59 * mm]
    else:
        widths = [170 * mm / column_count] * column_count
    table = Table(rows, repeatRows=1, hAlign="LEFT", colWidths=widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F2994A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "CabinBold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#EDC49F")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#FFF3E8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def figure_for(marker: str, figure_dir: Path = FIGURE_DIR):
    if marker.startswith("Figure 3.1"):
        return ArchitectureDiagram(), "Figure 3.1 - CabInspector multimodal architecture."
    if marker.startswith("Figure 3.2"):
        return DecisionFlowDiagram(), "Figure 3.2 - Evidence, privacy, and alert decision flow."
    if marker.startswith("Figure 4.1"):
        return DashboardDiagram(), "Figure 4.1 - CabInspector dashboard layout."
    if marker.startswith("Figure 4.2"):
        if not COMBINED_ACCEPTANCE_SCREENSHOT.is_file():
            raise FileNotFoundError(
                f"Combined acceptance screenshot not found: {COMBINED_ACCEPTANCE_SCREENSHOT}"
            )
        return (
            Image(str(COMBINED_ACCEPTANCE_SCREENSHOT), width=160 * mm, height=90 * mm),
            "Figure 4.2 - Historical combined-dashboard acceptance snapshot (14 September 2026; before the compact risk-summary text update).",
        )
    evaluation_figures = {
        "Figure 5.1": ("telemetry_confusion_matrix.png", "Figure 5.1 - PRIMUS argmax confusion matrix on continuous windows from held-out trips."),
        "Figure 5.2": ("telemetry_per_class_metrics.png", "Figure 5.2 - PRIMUS argmax per-category precision, recall, and F1 across held-out trips."),
        "Figure 5.3": ("telemetry_model_comparison.png", "Figure 5.3 - Trip-held-out comparison at the current 0.55 replay threshold, including always-NORMAL."),
        "Figure 5.4": ("telemetry_threshold_sensitivity.png", "Figure 5.4 - Descriptive confidence-threshold sensitivity; no new threshold is selected."),
        "Figure 5.5": ("fusion_weight_sensitivity.png", "Figure 5.5 - Synthetic fusion-weight sensitivity."),
        "Figure 5.6": ("fusion_baseline_ablation.png", "Figure 5.6 - Unconstrained and dependency-aware synthetic fusion comparisons and score ablations; not accuracy."),
        "Figure 5.7": ("combined_resource_summary.png", "Figure 5.7 - Representative 900-frame bounded combined-run CPU, memory, GPU, and VRAM summary."),
    }
    for figure_marker, (filename, caption) in evaluation_figures.items():
        if marker.startswith(figure_marker):
            path = Path(figure_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Evaluation figure not found: {path}")
            if filename == "telemetry_model_comparison.png":
                image = Image(str(path), width=170 * mm, height=70 * mm)
            elif filename == "fusion_baseline_ablation.png":
                image = Image(str(path), width=170 * mm, height=103 * mm)
            elif filename == "combined_resource_summary.png":
                image = Image(str(path), width=170 * mm, height=88 * mm)
            elif filename == "telemetry_per_class_metrics.png":
                image = Image(str(path), width=170 * mm, height=100 * mm)
            elif filename == "telemetry_threshold_sensitivity.png":
                image = Image(str(path), width=170 * mm, height=83 * mm)
            else:
                image = Image(str(path), width=170 * mm, height=112 * mm)
            return image, caption
    return None


def build_story(markdown: str, style_map, figure_dir: Path = FIGURE_DIR):
    story = []
    lines = markdown.splitlines()
    buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []
    index = 0

    def flush_paragraph():
        nonlocal buffer
        if buffer:
            text = " ".join(item.strip() for item in buffer)
            if not text.startswith("> ") and not text.startswith("*Chapter "):
                story.append(Paragraph(inline_markup(text), style_map["body"]))
            buffer = []

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), style_map["code"]))
                code_lines = []
            in_code = not in_code
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if (
            stripped.startswith("**Table 3.3")
            or stripped.startswith("**Table 5.3")
            or stripped.startswith("**Table 5.4")
            or stripped.startswith("**Table 5.5")
            or stripped.startswith("**Table 5.6")
        ):
            flush_paragraph()
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            if table_lines:
                standalone_caption = bool(re.fullmatch(r"\*\*Table .*\*\*\.?", stripped))
                caption = (
                    re.sub(r"^\*\*|\*\*$", "", stripped)
                    if standalone_caption
                    else stripped
                )
                story.append(
                    KeepTogether(
                        [
                            Spacer(1, 3),
                            Paragraph(inline_markup(caption), style_map["caption"]),
                            make_table(table_lines, style_map),
                            Spacer(1, 8),
                        ]
                    )
                )
            else:
                story.append(Paragraph(inline_markup(stripped), style_map["body"]))
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(Spacer(1, 3))
            story.append(make_table(table_lines, style_map))
            story.append(Spacer(1, 8))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            heading = stripped[2:].strip()
            if heading == "CabInspector" or heading.startswith("CabInspector:"):
                index += 1
                continue
            if heading.startswith("Chapter"):
                if story:
                    story.append(PageBreak())
                story.append(Paragraph(inline_markup(heading), style_map["chapter"]))
            else:
                story.append(PageBreak())
                story.append(Paragraph(inline_markup(heading), style_map["chapter"]))
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), style_map["section"]))
            index += 1
            continue
        if stripped.startswith("**Figure "):
            flush_paragraph()
            marker = re.sub(r"\*", "", stripped)
            figure = figure_for(marker, figure_dir)
            if figure:
                diagram, caption = figure
                # Keep report figures with their captions. Evaluation and
                # implementation figures flow naturally instead of forcing one
                # mostly-empty page per image.
                if marker.startswith("Figure 5.") or marker.startswith("Figure 4."):
                    story.append(
                        KeepTogether(
                            [Spacer(1, 8), diagram, Paragraph(caption, style_map["caption"]), Spacer(1, 10)]
                        )
                    )
                else:
                    story.extend([PageBreak(), Spacer(1, 8), diagram, Paragraph(caption, style_map["caption"]), Spacer(1, 10)])
            else:
                story.append(Paragraph(inline_markup(marker), style_map["caption"]))
            index += 1
            image_index = index
            while image_index < len(lines) and not lines[image_index].strip():
                image_index += 1
            if image_index < len(lines) and re.fullmatch(
                r"!\[[^\]]*\]\([^)]*\)", lines[image_index].strip()
            ):
                index = image_index + 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(ListItem(Paragraph(inline_markup(lines[index].strip()[2:]), style_map["bullet"])))
                index += 1
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=18))
            story.append(Spacer(1, 6))
            continue
        if re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            items = []
            while index < len(lines) and re.match(r"^\d+\. ", lines[index].strip()):
                text = re.sub(r"^\d+\. ", "", lines[index].strip())
                items.append(ListItem(Paragraph(inline_markup(text), style_map["bullet"])))
                index += 1
            list_flow = ListFlowable(items, bulletType="1", leftIndent=22)
            if story and isinstance(story[-1], Paragraph) and story[-1].getPlainText().rstrip().endswith(":"):
                list_intro = story.pop()
                story.append(KeepTogether([list_intro, list_flow, Spacer(1, 6)]))
            else:
                story.append(list_flow)
            story.append(Spacer(1, 6))
            continue
        if stripped.startswith("[" ) and "] " in stripped and story and any(getattr(item, "text", "") for item in story[-1:]):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped), style_map["reference"]))
            index += 1
            continue
        if stripped.startswith("[Student:") or stripped.startswith("[Programme:") or stripped.startswith("[Module") or stripped.startswith("**Ahmed Hossam") or stripped.startswith("**Student:") or stripped.startswith("**Programme:") or stripped.startswith("**Module") or stripped.startswith("**Report status"):
            index += 1
            continue
        buffer.append(stripped)
        index += 1
    flush_paragraph()
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CabInspector project report PDF.")
    parser.add_argument(
        "--report",
        type=Path,
        default=SOURCE,
        help="Editable Markdown report source (default: docs/report_revision/CabInspector_Final_Report_Revised.md).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory containing report evaluation figures.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Destination PDF path (default: output/pdf/CabInspector_Final_Report.pdf).",
    )
    args = parser.parse_args()
    output_path = args.output.resolve()
    source_path = args.report.resolve()
    figure_dir = args.figure_dir.resolve()
    register_fonts()
    if not source_path.is_file():
        raise FileNotFoundError(f"Report source not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = source_path.read_text(encoding="utf-8")
    style_map = styles()
    story = [
        Spacer(1, 48 * mm),
        Paragraph("CabInspector", style_map["title"]),
        Spacer(1, 28 * mm),
        Paragraph("CM3020 - AI 4.1: Orchestrating AI models to achieve a goal", style_map["subtitle"]),
        Spacer(1, 8 * mm),
        Paragraph("Ahmed Hossam", style_map["subtitle"]),
        PageBreak(),
    ]
    story.extend(build_story(text, style_map, figure_dir))
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title="CabInspector - Final Project Report",
        author="Ahmed Hossam",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(output_path)


if __name__ == "__main__":
    main()
