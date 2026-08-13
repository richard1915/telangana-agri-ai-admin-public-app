"""
Farmer/agronomist PDF report generator.

Produces a downloadable PDF combining: soil information, weather,
crop recommendation, MOA (optimizer) recommendation, estimated cost
savings, environmental impact, and a simple location map -- everything
a field visit or agronomist review needs in one document instead of
screenshots of the app.

Uses reportlab (already a project dependency via requirements.txt after
this change) so no external services or internet access is required to
draw the location marker -- it's rendered directly as vector graphics,
not a fetched map tile.
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.graphics.shapes import Drawing, Rect, Circle, String, Line
from reportlab.graphics import renderPDF

DARK_GREEN = colors.HexColor("#2e7d32")
BROWN = colors.HexColor("#6d4c30")
LIGHT_GREEN_BG = colors.HexColor("#f4f7f1")
CREAM = colors.HexColor("#f4ede3")


def _location_marker_drawing(latitude, longitude, width=400, height=200):
    """
    Lightweight vector 'map' -- a soil-toned grid with the farm location
    marked at its center and coordinates labeled. Not a real map tile
    (no network access at report-generation time); intended as a quick
    visual locator inside the PDF, consistent with the app's dark
    green / brown theme.
    """
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=CREAM, strokeColor=BROWN, strokeWidth=1))

    # Simple grid lines for a "map-like" feel.
    for x in range(0, width, 40):
        d.add(Line(x, 0, x, height, strokeColor=colors.HexColor("#d8c9b0"), strokeWidth=0.4))
    for y in range(0, height, 40):
        d.add(Line(0, y, width, y, strokeColor=colors.HexColor("#d8c9b0"), strokeWidth=0.4))

    cx, cy = width / 2, height / 2
    d.add(Circle(cx, cy, 7, fillColor=DARK_GREEN, strokeColor=colors.white, strokeWidth=1.5))
    d.add(Circle(cx, cy, 14, fillColor=None, strokeColor=DARK_GREEN, strokeWidth=1))
    label = f"{latitude:.4f}, {longitude:.4f}" if latitude is not None and longitude is not None else "Location not recorded"
    d.add(String(cx, cy - 26, label, fontSize=9, fillColor=BROWN, textAnchor="middle"))
    d.add(String(cx, height - 16, "Farm location (approximate)", fontSize=8, fillColor=BROWN, textAnchor="middle"))
    return d


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=18, leading=22, textColor=DARK_GREEN, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=10, leading=14, textColor=BROWN, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=13, leading=16, textColor=colors.white,
        backColor=DARK_GREEN, spaceBefore=14, spaceAfter=8, leftIndent=6, borderPadding=4,
    ))
    styles.add(ParagraphStyle(
        name="Note", fontSize=8, leading=11, textColor=colors.HexColor("#5a4a3a"),
    ))
    return styles


def _kv_table(rows, col_widths=(6 * cm, 9 * cm)):
    data = [[str(k), str(v)] for k, v in rows]
    t = Table(data, colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), BROWN),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREEN_BG]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde5d6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde5d6")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def generate_pdf_report(output_path, context):
    """
    Args:
        output_path: where to write the PDF
        context: dict with keys (all optional -- missing sections are
            simply skipped):
              farmer: {farmer_name, district, village, farm_size, phone}
              soil: {soil_type, ph, moisture, organic_carbon, temperature,
                     nitrogen, phosphorus, potassium, electrical_conductivity}
              weather: {temperature, rainfall, humidity, condition, ...} (free-form)
              crop_recommendation: {next_crop, recommended_chemical,
                     predicted_yield, confidence_score, limiting_factor}
              moa_recommendation: {method, initial_dose, optimized_dose,
                     reduction_percentage, farm_area}
              cost_savings: output of backend.cost_savings.estimate_cost_savings
              environmental_impact: output of backend.environmental_impact.estimate_environmental_impact
              latitude, longitude: floats
              generated_by: student/agronomist name

    Returns output_path.
    """
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
    )
    story = []

    farmer = context.get("farmer", {})
    soil = context.get("soil", {})
    weather = context.get("weather", {})
    crop_rec = context.get("crop_recommendation", {})
    moa_rec = context.get("moa_recommendation", {})
    cost = context.get("cost_savings", {})
    env = context.get("environmental_impact", {})

    story.append(Paragraph("Telangana Smart Agriculture AI -- Farm Advisory Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} "
        f"by {context.get('generated_by', 'Not specified')} | Advisory purpose only -- "
        f"designed to reduce unnecessary chemical use, subject to agronomist review.",
        styles["ReportSubtitle"],
    ))

    if farmer:
        story.append(Paragraph("Farmer & Farm Details", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Farmer name", farmer.get("farmer_name", "-")),
            ("District / Village", f"{farmer.get('district', '-')} / {farmer.get('village', '-')}"),
            ("Farm size (acres)", farmer.get("farm_size", "-")),
            ("Phone", farmer.get("phone", "-")),
        ]))
        story.append(Spacer(1, 6))
        story.append(_location_marker_drawing(context.get("latitude"), context.get("longitude")))

    if soil:
        story.append(Paragraph("Soil Information", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Soil type", soil.get("soil_type", "-")),
            ("pH", soil.get("ph", "-")),
            ("Moisture (%)", soil.get("moisture", "-")),
            ("Organic carbon (%)", soil.get("organic_carbon", "-")),
            ("Nitrogen (kg/acre)", soil.get("nitrogen", "-")),
            ("Phosphorus (kg/acre)", soil.get("phosphorus", "-")),
            ("Potassium (kg/acre)", soil.get("potassium", "-")),
            ("Electrical conductivity (dS/m)", soil.get("electrical_conductivity", "-")),
        ]))

    if weather:
        story.append(Paragraph("Weather", styles["SectionHeading"]))
        story.append(_kv_table([(k.replace("_", " ").title(), v) for k, v in weather.items()]))

    if crop_rec:
        story.append(Paragraph("Crop & Chemical Recommendation", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Recommended next crop", crop_rec.get("next_crop", "-")),
            ("Recommended chemical", crop_rec.get("recommended_chemical", "-")),
            ("Predicted yield (kg/acre)", crop_rec.get("predicted_yield", "-")),
            ("Model confidence (%)", crop_rec.get("confidence_score", "-")),
            ("Limiting factor", crop_rec.get("limiting_factor", "-")),
        ]))

    if moa_rec:
        story.append(Paragraph("MOA / Optimizer Recommendation", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Method", moa_rec.get("method", "MOA (Meerkat Optimization Algorithm)")),
            ("Initial dose (kg/acre)", moa_rec.get("initial_dose", "-")),
            ("Optimized dose (kg/acre)", moa_rec.get("optimized_dose", "-")),
            ("Reduction (%)", moa_rec.get("reduction_percentage", "-")),
            ("Farm area (acres)", moa_rec.get("farm_area", "-")),
        ]))

    if cost:
        story.append(Paragraph("Estimated Cost Savings", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Chemical", cost.get("chemical_name", "-")),
            ("₹ saved per acre", cost.get("rupees_saved_per_acre", "-")),
            ("₹ saved (whole farm)", cost.get("rupees_saved_total", "-")),
            ("Chemical saved (kg/acre)", cost.get("kg_saved_per_acre", "-")),
            ("Chemical saved (kg, whole farm)", cost.get("kg_saved_total", "-")),
            ("Percentage reduction", f"{cost.get('percentage_reduction', '-')}%"),
        ]))
        if cost.get("price_is_indicative"):
            story.append(Paragraph(
                "Note: exact market price for this chemical was not on file -- an indicative "
                "default price was used. Treat rupee figures as estimates.", styles["Note"],
            ))

    if env:
        story.append(Paragraph("Environmental Impact", styles["SectionHeading"]))
        story.append(_kv_table([
            ("Chemical reduction (%)", env.get("chemical_reduction_pct", "-")),
            ("Estimated residue reduction (%)", env.get("estimated_residue_reduction_pct", "-")),
            ("Soil health improvement indicator", env.get("soil_health_improvement_indicator", "-")),
        ]))
        story.append(Paragraph(env.get("note", ""), styles["Note"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This report is an AI-assisted decision-support estimate. It is designed to reduce "
        "unnecessary chemical use and should be reviewed by a qualified agronomist/AEO before "
        "field application, per standard operating procedure.", styles["Note"],
    ))

    doc.build(story)
    return output_path
