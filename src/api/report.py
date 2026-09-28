"""PDF reports from the exact prediction returned to the user."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from src.api.localization import translate, class_label

FONT = Path(__file__).resolve().parents[2] / "assets/fonts/NotoSansDevanagari-Full.ttf"
pdfmetrics.registerFont(TTFont("NotoHindi", str(FONT), shapable=True))
pdfmetrics.registerFontFamily("NotoHindi", normal="NotoHindi", bold="NotoHindi", italic="NotoHindi", boldItalic="NotoHindi")


def create_report(result, original, heatmap, destination, language="en"):
    styles = getSampleStyleSheet()
    t = lambda value: translate(value, language)
    if language == "hi":
        for style in styles.byName.values():
            if not hasattr(style, "fontSize"):
                continue
            style.fontName = "NotoHindi"
            style.shaping = True
            style.leading = max(style.leading, style.fontSize * 1.5)
    styles["Title"].textColor = colors.HexColor("#166534")
    styles["Title"].alignment = 0
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 16 if language == "hi" else 14

    def p(value, style="BodyText"):
        return Paragraph(escape(t(value)), styles[style])

    def photo(path):
        with PILImage.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((800, 600))
            width, height = image.size
            data = BytesIO()
            image.save(data, format="JPEG", quality=90)
        data.seek(0)
        scale = min(225 / width, 145 / height)
        return Image(data, width=width * scale, height=height * scale)

    story = [p("AgriVisionAI", "Title"), p("LEAF ANALYSIS REPORT", "Heading2"),
             p(t("Generated") + ": " + result["generated_at"]), p(t("Report ID") + ": " + result["report_id"]),
             Spacer(1, 10), p(class_label(result["disease"], language), "Heading2"),
             p(t("Leading prediction confidence") + f": {result['confidence']}%"), Spacer(1, 10)]
    images = Table([[photo(original), photo(heatmap)], [p("Uploaded leaf"), p("Model attention heatmap")]], colWidths=[245, 245])
    images.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [images]
    if result.get("uncertain"):
        story.append(p("Uncertain result - compare alternatives and seek confirmation.", "Heading3"))
    for warning in result.get("warnings", []):
        story.append(p(warning))
    story.append(p("Top 3 predictions", "Heading2"))
    rows = [[p("Class"), p("Confidence")]]
    for item in result["top_predictions"]:
        rows.append([p(class_label(item["disease"], language)), p(f"{item['confidence']}%")])
    table = Table(rows, colWidths=[390, 100], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcfce7")),
                              ("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    story += [table, Spacer(1, 8), p("Scores are estimates across all supported classes; the top three need not total 100%.")]
    care = [[p(key.title(), "Heading3"), p(result.get(key, "Not available"))]
            for key in ("symptoms", "cause", "treatment", "prevention")]
    care_table = Table([[care[0], care[1]], [care[2], care[3]]], colWidths=[245, 245])
    care_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 14)]))
    story.append(care_table)
    story += [Spacer(1, 10), p("This is not a confirmed diagnosis. Unsupported photos can still receive confident predictions. Confirm with a local agricultural expert before treatment. Heatmaps show model attention, not a measured diseased area.")]

    def footer(canvas, doc):
        line = p("AgriVisionAI | " + t("Model-generated analysis") + " | " + t("Page") + f" {doc.page}")
        line.wrap(A4[0] - 84, 30)
        line.drawOn(canvas, 42, 20)

    SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=36, bottomMargin=42, title="AgriVisionAI leaf analysis").build(story, onFirstPage=footer, onLaterPages=footer)
