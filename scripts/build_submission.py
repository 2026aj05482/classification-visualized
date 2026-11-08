"""Create a printable PDF from the complete README, keeping one source of truth."""

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "submission.pdf"
INK = colors.HexColor("#142B3B")
TEAL = colors.HexColor("#087F8C")
WIDTH = A4[0] - 34 * mm


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#087F8C">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(TEAL)
    canvas.line(17 * mm, 16 * mm, A4[0] - 17 * mm, 16 * mm)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(17 * mm, 11 * mm, "Classification Visualized | Machine Learning Assignment 2")
    canvas.drawRightString(A4[0] - 17 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    base = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9, leading=13, textColor=INK, spaceAfter=7),
        "h1": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=29,
                             textColor=INK, alignment=TA_LEFT, spaceAfter=10),
        "h2": ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=TEAL,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True),
        "h3": ParagraphStyle("Subsection", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=INK,
                             spaceBefore=10, spaceAfter=7, keepWithNext=True),
        "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8, leading=11, textColor=INK),
        "head": ParagraphStyle("Head", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=colors.white),
        "code": ParagraphStyle("Code", fontName="Courier", fontSize=7.3, leading=10, textColor=INK,
                               backColor=colors.HexColor("#EFF4F7"), borderPadding=8, spaceAfter=10),
    }
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code), styles["code"]))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            count = len(rows[0])
            widths = [WIDTH * .31] + [WIDTH * .69 / (count - 1)] * (count - 1) if count == 7 else [WIDTH * .26, WIDTH * .74]
            cells = [[Paragraph(inline(cell), styles["head" if r == 0 else "cell"]) for cell in row] for r, row in enumerate(rows)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EFF4F7")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#C4D4DE")),
            ]))
            story.extend([table, Spacer(1, 8)])
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            story.append(Paragraph(inline(line.lstrip("# ")), styles[f"h{level}"]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=8))
            index += 1
            continue
        paragraph = [line]
        index += 1
        if re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), styles["body"]))
            continue
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "```")) and not re.match(r"^\d+\. ", lines[index]):
            paragraph.append(lines[index])
            index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
                            topMargin=17 * mm, bottomMargin=23 * mm,
                            title="Classification Visualized - Assignment 2", author="2026aj05482")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
