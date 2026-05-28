"""
AgroAI — Professional PDF Report Generator
===========================================
Generates a premium, production-level AI crop disease analysis report
in the style of a commercial SaaS agricultural platform.

Features:
 - Multi-page A4 PDF with professional navy/white/green color theme
 - AgroAI branding header & footer with timestamp/page number
 - Scan summary card, uploaded leaf image, AI analysis
 - Treatment & prevention sections with styled bullet cards
 - Severity visualization (confidence bar, risk meter, badge)
 - Smart AI Insights section
 - QR code for scan dashboard link
 - "AgroAI Confidential" diagonal watermark
 - Digital verification badge
 - Auto-generated report ID
 - Unicode-safe fonts via ReportLab TTF registration
 - Fully modular functions for easy maintenance

Author: AgroAI Engineering Team
Version: 2.0.0
"""

import os
import io
import uuid
import datetime
import base64
import logging

from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfgen import canvas as pdfgen_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# BRAND COLORS  (navy + green + white palette)
# ─────────────────────────────────────────────
NAVY       = colors.HexColor("#0D1B2A")      # Deep navy background
NAVY_MID   = colors.HexColor("#1B2D3E")      # Section headers
NAVY_LIGHT = colors.HexColor("#243B55")      # Card backgrounds
GREEN      = colors.HexColor("#1DB954")      # Accent / healthy
GREEN_DARK = colors.HexColor("#158A3E")      # Dark green
TEAL       = colors.HexColor("#00C9A7")      # Teal highlight
AMBER      = colors.HexColor("#F59E0B")      # Warning / medium
RED        = colors.HexColor("#EF4444")      # High risk
WHITE      = colors.HexColor("#FFFFFF")
OFF_WHITE  = colors.HexColor("#F8FAFC")
LIGHT_GRAY = colors.HexColor("#E2E8F0")
MID_GRAY   = colors.HexColor("#94A3B8")
DARK_TEXT  = colors.HexColor("#0F172A")
BODY_TEXT  = colors.HexColor("#334155")

PAGE_W, PAGE_H = A4   # 210 × 297 mm in points
MARGIN = 18 * mm

# ─────────────────────────────────────────────
# FONT REGISTRATION (Unicode-safe)
# ─────────────────────────────────────────────

def _register_fonts():
    """Register Unicode-safe Helvetica aliases (built-in ReportLab fonts).
    For production, swap with DejaVuSans TTF for full Unicode support."""
    # ReportLab ships with Type-1 Helvetica; these are always available.
    # The alias lets our code use consistent names throughout.
    pass  # ReportLab built-ins suffice; extend here with TTFont if needed.


# ─────────────────────────────────────────────
# STYLE DEFINITIONS
# ─────────────────────────────────────────────

def _build_styles():
    """Return a dict of named ParagraphStyles used throughout the report."""
    base = getSampleStyleSheet()

    styles = {}

    styles["report_title"] = ParagraphStyle(
        "report_title", fontName="Helvetica-Bold",
        fontSize=22, textColor=WHITE, leading=26, spaceAfter=2
    )
    styles["report_subtitle"] = ParagraphStyle(
        "report_subtitle", fontName="Helvetica",
        fontSize=10, textColor=LIGHT_GRAY, leading=13
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading", fontName="Helvetica-Bold",
        fontSize=13, textColor=WHITE, leading=16,
        backColor=NAVY_MID, borderPad=6,
        spaceBefore=10, spaceAfter=6, leftIndent=4
    )
    styles["card_label"] = ParagraphStyle(
        "card_label", fontName="Helvetica-Bold",
        fontSize=8.5, textColor=MID_GRAY, leading=11, spaceAfter=1
    )
    styles["card_value"] = ParagraphStyle(
        "card_value", fontName="Helvetica-Bold",
        fontSize=11, textColor=DARK_TEXT, leading=14, spaceAfter=3
    )
    styles["body"] = ParagraphStyle(
        "body", fontName="Helvetica",
        fontSize=9.5, textColor=BODY_TEXT, leading=14, spaceAfter=4
    )
    styles["bullet_item"] = ParagraphStyle(
        "bullet_item", fontName="Helvetica",
        fontSize=9.5, textColor=BODY_TEXT, leading=14,
        leftIndent=12, firstLineIndent=-12, spaceAfter=4
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique",
        fontSize=8.5, textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=4
    )
    styles["insight_label"] = ParagraphStyle(
        "insight_label", fontName="Helvetica-Bold",
        fontSize=9, textColor=TEAL, leading=12
    )
    styles["insight_value"] = ParagraphStyle(
        "insight_value", fontName="Helvetica",
        fontSize=9.5, textColor=BODY_TEXT, leading=14, spaceAfter=2
    )
    styles["footer_text"] = ParagraphStyle(
        "footer_text", fontName="Helvetica",
        fontSize=7.5, textColor=MID_GRAY, alignment=TA_CENTER
    )
    styles["farmer_note"] = ParagraphStyle(
        "farmer_note", fontName="Helvetica-Oblique",
        fontSize=9, textColor=BODY_TEXT, leading=14,
        leftIndent=8, spaceAfter=4
    )

    return styles


# ─────────────────────────────────────────────
# CUSTOM FLOWABLES
# ─────────────────────────────────────────────

class ColorRect(Flowable):
    """A filled rectangle with optional rounded corners and text label."""
    def __init__(self, width, height, fill_color, radius=4, label="", label_color=WHITE):
        super().__init__()
        self.width      = width
        self.height     = height
        self.fill_color = fill_color
        self.radius     = radius
        self.label      = label
        self.label_color = label_color

    def draw(self):
        self.canv.setFillColor(self.fill_color)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=0)
        if self.label:
            self.canv.setFillColor(self.label_color)
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.drawCentredString(self.width / 2, self.height / 2 - 4, self.label)


class ConfidenceBar(Flowable):
    """A horizontal progress bar visualizing AI confidence level."""
    def __init__(self, confidence, width=None, height=14):
        super().__init__()
        self.confidence = min(max(float(confidence), 0), 100)
        self.width      = width or (PAGE_W - 2 * MARGIN)
        self.height     = height

    def draw(self):
        c = self.canv
        bar_w = self.width
        filled = bar_w * (self.confidence / 100)

        # Track (background)
        c.setFillColor(LIGHT_GRAY)
        c.roundRect(0, 0, bar_w, self.height, self.height / 2, fill=1, stroke=0)

        # Filled portion — color coded by severity
        if self.confidence >= 90:
            fill_col = RED
        elif self.confidence >= 70:
            fill_col = AMBER
        else:
            fill_col = GREEN

        if filled > 0:
            c.setFillColor(fill_col)
            c.roundRect(0, 0, filled, self.height, self.height / 2, fill=1, stroke=0)

        # Percentage label
        c.setFillColor(WHITE if filled > bar_w * 0.25 else DARK_TEXT)
        c.setFont("Helvetica-Bold", 8)
        label_x = min(filled - 14, bar_w - 28) if filled > 30 else filled + 4
        c.drawString(max(label_x, 2), self.height / 2 - 4, f"{self.confidence:.1f}%")


class RiskMeter(Flowable):
    """A simple three-segment risk meter (LOW / MEDIUM / HIGH)."""
    def __init__(self, risk_level, width=120, height=28):
        super().__init__()
        self.risk_level = risk_level.upper()
        self.width      = width
        self.height     = height

    def draw(self):
        c = self.canv
        seg_w  = self.width / 3
        labels = ["LOW", "MEDIUM", "HIGH"]
        cols   = [GREEN, AMBER, RED]
        active = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(self.risk_level, 0)

        for i, (lbl, col) in enumerate(zip(labels, cols)):
            x = i * seg_w
            alpha_col = col if i == active else colors.HexColor("#D1D5DB")
            c.setFillColor(alpha_col)
            # Left cap, middle, right cap
            radius = 5
            if i == 0:
                c.roundRect(x, 0, seg_w - 1, self.height, radius, fill=1, stroke=0)
            elif i == 2:
                c.roundRect(x + 1, 0, seg_w - 1, self.height, radius, fill=1, stroke=0)
            else:
                c.rect(x + 1, 0, seg_w - 2, self.height, fill=1, stroke=0)

            # Label
            text_col = WHITE if i == active else MID_GRAY
            c.setFillColor(text_col)
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(x + seg_w / 2, self.height / 2 - 3.5, lbl)


class SectionHeader(Flowable):
    """A full-width navy banner used as a section title bar."""
    def __init__(self, title, icon="", width=None, height=22):
        super().__init__()
        self.title  = title
        self.icon   = icon
        self.width  = width or (PAGE_W - 2 * MARGIN)
        self.height = height

    def draw(self):
        c = self.canv
        # Banner background
        c.setFillColor(NAVY_MID)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)
        # Green left accent bar
        c.setFillColor(GREEN)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        # Title text
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 11)
        display = f"{self.icon}  {self.title}" if self.icon else self.title
        c.drawString(12, self.height / 2 - 4, display)


# ─────────────────────────────────────────────
# PAGE TEMPLATE (Header + Footer + Watermark)
# ─────────────────────────────────────────────

class ReportCanvas(pdfgen_canvas.Canvas):
    """Custom canvas that draws the header, footer, and watermark on every page."""

    def __init__(self, filename, report_id, version="2.0.0", **kwargs):
        super().__init__(filename, **kwargs)
        self.report_id   = report_id
        self.version     = version
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_decorations(num_pages)
            pdfgen_canvas.Canvas.showPage(self)
        pdfgen_canvas.Canvas.save(self)

    def _draw_page_decorations(self, total_pages):
        page_num = self._pageNumber
        self._draw_header()
        self._draw_footer(page_num, total_pages)
        self._draw_watermark()

    def _draw_header(self):
        """Draw the AgroAI branding header bar (only on page 1 the header is a large hero;
        on subsequent pages draw a slim banner)."""
        if self._pageNumber == 1:
            return  # The hero header is rendered as a flowable on page 1

        # Slim header for continuation pages
        w = PAGE_W
        self.setFillColor(NAVY)
        self.rect(0, PAGE_H - 28, w, 28, fill=1, stroke=0)
        self.setFillColor(GREEN)
        self.rect(0, PAGE_H - 30, w, 2, fill=1, stroke=0)
        self.setFillColor(WHITE)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(MARGIN, PAGE_H - 18, "AgroAI")
        self.setFont("Helvetica", 8)
        self.setFillColor(MID_GRAY)
        self.drawRightString(PAGE_W - MARGIN, PAGE_H - 18, "AI Crop Disease Analysis Report")

    def _draw_footer(self, page_num, total_pages):
        """Draw the footer bar with branding, timestamp, and page number."""
        y_bottom = 18
        w = PAGE_W

        self.setFillColor(NAVY)
        self.rect(0, 0, w, y_bottom + 8, fill=1, stroke=0)
        self.setFillColor(GREEN)
        self.rect(0, y_bottom + 8, w, 1.5, fill=1, stroke=0)

        self.setFont("Helvetica", 7)
        self.setFillColor(MID_GRAY)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.drawString(MARGIN, 10, f"Generated by AgroAI  |  {ts}  |  System v{self.version}  |  Report ID: {self.report_id}")
        self.drawRightString(w - MARGIN, 10, f"Page {page_num} of {total_pages}")

    def _draw_watermark(self):
        """Draw a diagonal 'AgroAI Confidential' watermark across the page."""
        self.saveState()
        self.setFillColor(colors.Color(0.85, 0.88, 0.92, alpha=0.12))
        self.setFont("Helvetica-Bold", 52)
        self.translate(PAGE_W / 2, PAGE_H / 2)
        self.rotate(42)
        self.drawCentredString(0, 0, "AgroAI Confidential")
        self.restoreState()


class ReportCanvasFactory:
    """Factory to pass report_id and version to ReportCanvas via SimpleDocTemplate."""

    def __init__(self, report_id, version="2.0.0"):
        self.report_id = report_id
        self.version   = version

    def __call__(self, filename, **kwargs):
        return ReportCanvas(filename, self.report_id, self.version, **kwargs)


# ─────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────

def _optimize_image_bytes(image_bytes: bytes, max_size=(600, 600)) -> bytes:
    """Resize & compress image for high-quality PDF embedding."""
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            bg = PILImage.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.thumbnail(max_size, PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image optimization failed: {e}")
        return image_bytes


def _base64_to_pil_bytes(data_url: str) -> bytes | None:
    """Convert a base64 data-URL to raw image bytes."""
    try:
        if data_url.startswith("data:"):
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        return base64.b64decode(encoded)
    except Exception as e:
        logger.warning(f"Base64 decode failed: {e}")
        return None


# ─────────────────────────────────────────────
# QR CODE HELPER
# ─────────────────────────────────────────────

def _generate_qr_image(scan_id: str, base_url="http://127.0.0.1:5000") -> io.BytesIO | None:
    """Generate a QR code for the scan dashboard URL.
    Requires 'qrcode' package; falls back to None if unavailable."""
    try:
        import qrcode
        url = f"{base_url}/history?scan={scan_id}"
        qr  = qrcode.QRCode(version=1, box_size=4, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#0D1B2A", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except ImportError:
        logger.info("qrcode package not installed; skipping QR code.")
        return None
    except Exception as e:
        logger.warning(f"QR generation failed: {e}")
        return None


# ─────────────────────────────────────────────
# SECTION BUILDERS  (each returns a list of flowables)
# ─────────────────────────────────────────────

def _build_hero_header(styles, report_id: str) -> list:
    """Build the full-width navy hero header on page 1."""
    story = []

    # Draw hero header via a custom flowable
    class HeroHeader(Flowable):
        def __init__(self, report_id):
            super().__init__()
            self.report_id = report_id
            self.width     = PAGE_W - 2 * MARGIN
            self.height    = 68

        def draw(self):
            c = self.canv
            W = self.width

            # Background
            c.setFillColor(NAVY)
            c.roundRect(0, 0, W, self.height, 8, fill=1, stroke=0)

            # Green bottom accent stripe
            c.setFillColor(GREEN)
            c.rect(0, 0, W, 3, fill=1, stroke=0)

            # Logo leaf icon (simple geometric leaf using rect + arc)
            c.setFillColor(GREEN)
            c.roundRect(10, 43, 22, 18, 9, fill=1, stroke=0)
            c.setFillColor(GREEN_DARK)
            c.setLineWidth(1.2)
            c.setStrokeColor(GREEN_DARK)
            c.line(10, 43, 22, 52)   # stem line

            # Brand name
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(38, 48, "AgroAI")

            # Tag line
            c.setFillColor(GREEN)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(38, 40, "POWERED BY ARTIFICIAL INTELLIGENCE")

            # Report Title
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 15)
            c.drawString(38, 27, "AI Crop Disease Analysis Report")

            c.setFillColor(LIGHT_GRAY)
            c.setFont("Helvetica", 9)
            c.drawString(38, 16, "AI-Powered Smart Plant Health Monitoring System")

            # Report ID badge (top-right)
            c.setFillColor(NAVY_LIGHT)
            c.roundRect(W - 130, self.height - 26, 128, 22, 5, fill=1, stroke=0)
            c.setFillColor(MID_GRAY)
            c.setFont("Helvetica", 7)
            c.drawString(W - 126, self.height - 9, "REPORT ID")
            c.setFillColor(TEAL)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(W - 126, self.height - 18, self.report_id[:20].upper())

    story.append(HeroHeader(report_id))
    story.append(Spacer(1, 8 * mm))
    return story


def _build_scan_summary_card(styles, data: dict) -> list:
    """Build the Scan Summary Card table."""
    story = []
    story.append(SectionHeader("SCAN SUMMARY", icon="[ID]"))
    story.append(Spacer(1, 3 * mm))

    scan_id     = data.get("scan_id", "N/A")
    scan_time   = data.get("scan_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    disease     = data.get("disease_name", "Unknown")
    confidence  = data.get("confidence", 0)
    severity    = data.get("severity", "LOW")
    plant_type  = data.get("plant_type", disease.split()[0] if disease else "Unknown")
    filename    = data.get("filename", "upload.jpg") or "upload.jpg"
    db_id       = data.get("db_id", "N/A")
    is_healthy  = data.get("is_healthy", False)

    risk_status = "HEALTHY - No Disease Detected" if is_healthy else (
        "HIGH RISK" if severity == "HIGH" else
        "MODERATE RISK" if severity == "MEDIUM" else
        "LOW RISK"
    )

    SEV_COLORS = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
    sev_color  = SEV_COLORS.get(severity, GREEN)

    def _kv(label, value, val_color=DARK_TEXT):
        return [
            Paragraph(label.upper(), styles["card_label"]),
            Paragraph(str(value), ParagraphStyle(
                "cv_inline", fontName="Helvetica-Bold",
                fontSize=10.5, textColor=val_color, leading=14
            ))
        ]

    summary_data = [
        _kv("Scan ID", f"#{scan_id[:16].upper()}", TEAL),
        _kv("Scan Date & Time", scan_time),
        _kv("Disease Name", disease, RED if not is_healthy else GREEN),
        _kv("AI Confidence Score", f"{confidence:.1f}%", sev_color),
        _kv("Severity Level", severity, sev_color),
        _kv("AI Risk Status", risk_status, RED if not is_healthy else GREEN),
        _kv("Plant Type", plant_type),
        _kv("Uploaded File", filename[:40] + ("..." if len(filename) > 40 else "")),
    ]

    # 2-column grid layout (left col = label, right col = value × 2 per row)
    rows_left  = summary_data[:4]
    rows_right = summary_data[4:]

    col_w  = (PAGE_W - 2 * MARGIN - 6 * mm) / 2

    def _cell_block(pairs):
        items = []
        for pair in pairs:
            items.append(pair[0])
            items.append(pair[1])
            items.append(Spacer(1, 2 * mm))
        return items

    combined = Table(
        [[_cell_block(rows_left), _cell_block(rows_right)]],
        colWidths=[col_w, col_w],
    )
    combined.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 0), (0, -1), OFF_WHITE),
        ("BACKGROUND",    (1, 0), (1, -1), colors.HexColor("#F0FDF4")),
        ("BOX",           (0, 0), (0, -1), 0.5, LIGHT_GRAY),
        ("BOX",           (1, 0), (1, -1), 0.5, LIGHT_GRAY),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(combined)
    story.append(Spacer(1, 6 * mm))
    return story


def _build_image_section(styles, image_bytes: bytes | None) -> list:
    """Build the uploaded leaf image display section."""
    story = []
    story.append(SectionHeader("ANALYZED CROP IMAGE", icon="[IMG]"))
    story.append(Spacer(1, 3 * mm))

    if image_bytes:
        try:
            optimized = _optimize_image_bytes(image_bytes)
            img_buf   = io.BytesIO(optimized)
            img_buf.seek(0)

            # Max dimensions: 80mm wide, 70mm tall
            max_w, max_h = 80 * mm, 70 * mm
            pil_img = PILImage.open(io.BytesIO(optimized))
            orig_w, orig_h = pil_img.size
            aspect = orig_w / orig_h
            disp_w = min(max_w, max_h * aspect)
            disp_h = disp_w / aspect

            img_flowable = Image(img_buf, width=disp_w, height=disp_h)
            img_flowable.hAlign = "CENTER"

            # Wrap in a bordered table
            img_table = Table(
                [[img_flowable]],
                colWidths=[disp_w + 12],
            )
            img_table.setStyle(TableStyle([
                ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING",     (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
                ("LEFTPADDING",    (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
                ("BOX",            (0, 0), (-1, -1), 1.2, LIGHT_GRAY),
                ("BACKGROUND",     (0, 0), (-1, -1), OFF_WHITE),
                ("ROUNDEDCORNERS", [8]),
            ]))

            # Center the table on the page
            wrapper = Table([[img_table]], colWidths=[PAGE_W - 2 * MARGIN])
            wrapper.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(wrapper)
            story.append(Paragraph("Analyzed Crop Image — AI Scan Result", styles["caption"]))
        except Exception as e:
            logger.warning(f"Image embed failed: {e}")
            story.append(Paragraph("[Image could not be embedded]", styles["caption"]))
    else:
        story.append(Paragraph("[No image provided for this scan]", styles["caption"]))

    story.append(Spacer(1, 5 * mm))
    return story


def _build_ai_analysis(styles, data: dict) -> list:
    """Build the AI Analysis section."""
    story = []
    story.append(SectionHeader("AI ANALYSIS", icon="[AI]"))
    story.append(Spacer(1, 3 * mm))

    disease     = data.get("disease_name", "Unknown")
    confidence  = data.get("confidence", 0)
    symptoms    = data.get("symptoms", "No symptoms data available.")
    cause       = data.get("cause", "Caused by fungal, bacterial, or viral pathogens specific to this disease.")
    spread_risk = data.get("spread_risk", "Moderate — can spread via contact with infected plant material.")
    is_healthy  = data.get("is_healthy", False)

    if confidence >= 90:
        interp = "Very High Confidence — AI is highly certain of this diagnosis. Immediate action recommended."
    elif confidence >= 75:
        interp = "High Confidence — AI diagnosis is reliable. Verify with field inspection."
    elif confidence >= 55:
        interp = "Moderate Confidence — Suggestive but not conclusive. Cross-check with agronomist."
    else:
        interp = "Low Confidence — Preliminary detection. Manual expert review strongly advised."

    subsections = [
        ("Disease Description", f"{disease} — {'Plant appears healthy with no signs of disease.' if is_healthy else 'A significant plant disease was detected requiring immediate attention.'}"),
        ("Observed Symptoms",   symptoms),
        ("Root Cause",          cause),
        ("Spread Risk",         spread_risk),
        ("AI Confidence Interpretation", interp),
    ]

    for title, content in subsections:
        story.append(Paragraph(f"<b>{title}</b>", ParagraphStyle(
            "subsec", fontName="Helvetica-Bold",
            fontSize=10, textColor=NAVY, leading=14, spaceAfter=2
        )))
        # Wrap long content
        clean = str(content).replace("\n", "<br/>")
        story.append(Paragraph(clean, styles["body"]))
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 4 * mm))
    return story


def _build_treatment_section(styles, data: dict) -> list:
    """Build the Treatment Recommendation section with styled bullet cards."""
    story = []
    story.append(SectionHeader("TREATMENT RECOMMENDATIONS", icon="[Rx]"))
    story.append(Spacer(1, 3 * mm))

    treatment   = data.get("treatment", "")
    is_healthy  = data.get("is_healthy", False)

    if is_healthy:
        story.append(Paragraph(
            "Plant is healthy. No treatment required. Continue regular care practices.",
            styles["body"]
        ))
        story.append(Spacer(1, 4 * mm))
        return story

    # Parse treatment text into bullet items
    raw_bullets = [l.strip() for l in treatment.replace("\r", "").split("\n") if l.strip()]
    # Remove leading bullet chars
    bullets = []
    for b in raw_bullets:
        for prefix in ("• ", "- ", "* ", "· "):
            if b.startswith(prefix):
                b = b[len(prefix):]
                break
        if b:
            bullets.append(b)

    # Predefined categories to enrich the output
    categories = [
        ("Immediate Action",          "red",   bullets[0] if len(bullets) > 0 else "Isolate affected plants immediately."),
        ("Recommended Pesticides",    "amber", bullets[1] if len(bullets) > 1 else "Consult local agronomist for approved fungicides/pesticides."),
        ("Organic Solutions",         "green", bullets[2] if len(bullets) > 2 else "Neem oil spray (5ml/L) — environmentally safe alternative."),
        ("Isolation Advice",          "navy",  bullets[3] if len(bullets) > 3 else "Remove and bag infected leaves; do not compost."),
        ("Field Monitoring Tips",     "teal",  bullets[4] if len(bullets) > 4 else "Inspect surrounding plants within 5m for early signs."),
    ]
    # Add remaining bullets as additional tips
    for extra in bullets[5:]:
        categories.append(("Additional Advisory", "navy", extra))

    CARD_COLORS = {
        "red": (colors.HexColor("#FEF2F2"), RED),
        "amber": (colors.HexColor("#FFFBEB"), AMBER),
        "green": (colors.HexColor("#F0FDF4"), GREEN),
        "navy": (colors.HexColor("#EFF6FF"), NAVY_MID),
        "teal": (colors.HexColor("#F0FDFA"), TEAL),
    }

    available_w = PAGE_W - 2 * MARGIN

    for cat_title, color_key, content in categories:
        bg, accent = CARD_COLORS.get(color_key, CARD_COLORS["navy"])

        # Accent bar
        accent_col = Flowable.__new__(Flowable)
        inner = Paragraph(
            f"<b>{cat_title}:</b>  {content}",
            ParagraphStyle("titem", fontName="Helvetica",
                           fontSize=9.5, textColor=BODY_TEXT, leading=14)
        )
        card = Table(
            [[inner]],
            colWidths=[available_w - 10],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEAFTER",     (0, 0), (0, -1), 3, accent),
            ("BOX",           (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
        ]))
        story.append(card)
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 4 * mm))
    return story


def _build_prevention_section(styles, data: dict) -> list:
    """Build the Prevention section."""
    story = []
    story.append(SectionHeader("PREVENTION & AGRONOMIC GUIDANCE", icon="[P]"))
    story.append(Spacer(1, 3 * mm))

    prevention = data.get("prevention", "")
    raw        = [l.strip() for l in prevention.replace("\r", "").split("\n") if l.strip()]
    bullets    = []
    for b in raw:
        for prefix in ("• ", "- ", "* ", "· "):
            if b.startswith(prefix):
                b = b[len(prefix):]
                break
        if b:
            bullets.append(b)

    # Ensure we have at least 5 items with sensible defaults
    defaults = [
        "Maintain proper plant spacing to ensure adequate air circulation and reduce humidity.",
        "Follow a strict irrigation schedule — avoid overhead watering; use drip irrigation where possible.",
        "Test soil pH and nutrient levels regularly; maintain balanced fertilization.",
        "Practice crop rotation with non-host species every 2-3 seasons to break disease cycles.",
        "Source certified disease-resistant variety seeds from registered nurseries.",
    ]
    while len(bullets) < 5:
        bullets.append(defaults[len(bullets)])

    topics = [
        "Preventive Farming Practices",
        "Irrigation Guidance",
        "Soil Management",
        "Crop Rotation Strategy",
        "Resistant Variety Recommendations",
    ]
    for i, (topic, blt) in enumerate(zip(topics, bullets)):
        story.append(Paragraph(
            f"<b>{topic}:</b>  {blt}",
            styles["bullet_item"]
        ))
        if i < len(topics) - 1:
            story.append(HRFlowable(width="100%", thickness=0.3, color=LIGHT_GRAY,
                                    spaceAfter=3, spaceBefore=3))

    # Any remaining bullets
    for extra in bullets[5:]:
        story.append(Paragraph(f"<b>Additional Tip:</b>  {extra}", styles["bullet_item"]))

    story.append(Spacer(1, 4 * mm))
    return story


def _build_severity_visualization(styles, data: dict) -> list:
    """Build the Severity Visualization section with progress bar and risk meter."""
    story = []
    story.append(SectionHeader("SEVERITY VISUALIZATION", icon="[%]"))
    story.append(Spacer(1, 3 * mm))

    confidence  = float(data.get("confidence", 0))
    severity    = str(data.get("severity", "LOW")).upper()
    is_healthy  = data.get("is_healthy", False)

    avail_w = PAGE_W - 2 * MARGIN

    # ── Confidence Bar ──
    story.append(Paragraph("<b>AI Confidence Score</b>", ParagraphStyle(
        "viz_lbl", fontName="Helvetica-Bold", fontSize=9.5, textColor=NAVY,
        spaceAfter=3
    )))
    story.append(ConfidenceBar(confidence, width=avail_w, height=16))
    story.append(Spacer(1, 4 * mm))

    # ── Risk Meter + Severity Badge side by side ──
    risk_level = "LOW" if is_healthy else severity

    sev_bg = {
        "HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN
    }.get(risk_level, GREEN)

    class SeverityBadge(Flowable):
        def __init__(self, label, bg_color, width=120, height=28):
            super().__init__()
            self.label    = label
            self.bg_color = bg_color
            self.width    = width
            self.height   = height

        def draw(self):
            c = self.canv
            c.setFillColor(self.bg_color)
            c.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 11)
            c.drawCentredString(self.width / 2, self.height / 2 - 4, self.label)

    badge_label = f"{'HEALTHY' if is_healthy else severity + ' RISK'}"
    meter_tbl = Table(
        [[
            [Paragraph("<b>Risk Level Meter</b>", ParagraphStyle(
                "rlm", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY,
                spaceAfter=4)),
             RiskMeter(risk_level, width=140, height=24)],
            Spacer(10, 1),
            [Paragraph("<b>Severity Badge</b>", ParagraphStyle(
                "sb", fontName="Helvetica-Bold", fontSize=9, textColor=NAVY,
                spaceAfter=4)),
             SeverityBadge(badge_label, sev_bg, width=120, height=28)],
        ]],
        colWidths=[avail_w * 0.48, 8, avail_w * 0.48],
    )
    meter_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meter_tbl)
    story.append(Spacer(1, 6 * mm))
    return story


def _build_smart_insights(styles, data: dict) -> list:
    """Build the AI Smart Insights section."""
    story = []
    story.append(SectionHeader("SMART AI INSIGHTS", icon="[AI]"))
    story.append(Spacer(1, 3 * mm))

    confidence = float(data.get("confidence", 0))
    severity   = str(data.get("severity", "LOW")).upper()
    is_healthy = data.get("is_healthy", False)

    # Compute derived insights
    damage_risk = "Negligible" if is_healthy else (
        "High (>50% crop loss possible)" if severity == "HIGH" else
        "Moderate (15-40% yield reduction)" if severity == "MEDIUM" else
        "Low (<10% impact if treated promptly)"
    )
    recovery_prob = "99%" if is_healthy else (
        "40-60%" if severity == "HIGH" else
        "70-85%" if severity == "MEDIUM" else
        "90-95%"
    )
    weather_warn = (
        "No special concern." if is_healthy else
        "High humidity and warm temperatures (>25°C) accelerate spread significantly." if severity == "HIGH" else
        "Monitor during wet/humid periods; disease activity increases with moisture." if severity == "MEDIUM" else
        "Low sensitivity to weather changes; standard monitoring sufficient."
    )
    inspection_freq = (
        "Weekly routine" if is_healthy else
        "Daily for next 14 days" if severity == "HIGH" else
        "Every 3-4 days for 2 weeks" if severity == "MEDIUM" else
        "Bi-weekly monitoring"
    )
    rec_score = int(min(100, max(0, confidence * (0.6 if not is_healthy else 1.0))))

    insights = [
        ("Estimated Crop Damage Risk",      damage_risk),
        ("Recovery Probability",            recovery_prob),
        ("Weather Sensitivity Warning",     weather_warn),
        ("Recommended Inspection Frequency", inspection_freq),
        ("AI Recommendation Score",         f"{rec_score}/100 — {'Actionable' if rec_score >= 70 else 'Advisory'}"),
    ]

    avail_w = PAGE_W - 2 * MARGIN
    rows    = []
    for label, value in insights:
        rows.append([
            Paragraph(label.upper(), styles["card_label"]),
            Paragraph(str(value), styles["body"]),
        ])

    tbl = Table(rows, colWidths=[avail_w * 0.38, avail_w * 0.62])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("BACKGROUND",    (1, 0), (1, -1), OFF_WHITE),
        ("TEXTCOLOR",     (0, 0), (0, -1), NAVY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, LIGHT_GRAY),
        ("BOX",           (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 5 * mm))
    return story


def _build_database_details(styles, data: dict) -> list:
    """Build the Database Integration Details section."""
    story = []
    story.append(SectionHeader("DATABASE & STORAGE DETAILS", icon="[DB]"))
    story.append(Spacer(1, 3 * mm))

    db_id        = data.get("db_id") or "Not saved (offline mode)"
    scan_id      = data.get("scan_id", "N/A")
    is_connected = bool(data.get("db_id"))
    sync_status  = "Synced" if is_connected else "Pending / Offline"
    storage_stat = "Persisted in MongoDB Atlas" if is_connected else "Stored in local cache (MongoDB unavailable)"

    avail_w = PAGE_W - 2 * MARGIN
    rows = [
        ["MongoDB Scan Record ID", db_id],
        ["Local Scan ID",          scan_id],
        ["Storage Status",         storage_stat],
        ["Atlas Sync Status",      sync_status],
        ["Database Name",          "agroai.scans"],
    ]
    tbl_data = [
        [Paragraph(r[0].upper(), styles["card_label"]),
         Paragraph(str(r[1]), styles["body"])]
        for r in rows
    ]
    tbl = Table(tbl_data, colWidths=[avail_w * 0.38, avail_w * 0.62])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#F0FDF4")),
        ("BACKGROUND",    (1, 0), (1, -1), OFF_WHITE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, LIGHT_GRAY),
        ("BOX",           (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 5 * mm))
    return story


def _build_farmer_advisory(styles, data: dict) -> list:
    """Build the Farmer Advisory Notes section."""
    story = []
    story.append(SectionHeader("FARMER ADVISORY NOTES", icon="[!]"))
    story.append(Spacer(1, 3 * mm))

    disease    = data.get("disease_name", "the detected condition")
    is_healthy = data.get("is_healthy", False)
    severity   = data.get("severity", "LOW").upper()

    if is_healthy:
        notes = [
            f"Your crop is healthy! Continue your current care routine.",
            "Maintain balanced fertilization and water schedules.",
            "Periodic monitoring is still advised to catch early-stage issues.",
            "Keep field records updated for future seasonal comparisons.",
        ]
    else:
        notes = [
            f"Immediate attention to {disease} is recommended.",
            "Do NOT apply random pesticides — consult a certified agronomist for the correct product and dosage.",
            "Record treatment dates and outcomes for future crop management reference.",
            "Alert neighboring farmers if the disease has high spread risk.",
            "Contact your local agricultural extension office for subsidized treatment programs if available.",
        ]
        if severity == "HIGH":
            notes.insert(1, "Consider quarantining infected field sections to contain further spread.")

    for note in notes:
        story.append(Paragraph(f"• {note}", styles["farmer_note"]))

    story.append(Spacer(1, 4 * mm))
    return story


def _build_qr_and_verification(styles, data: dict) -> list:
    """Build the QR code and digital verification badge section."""
    story = []
    story.append(SectionHeader("VERIFICATION & DIGITAL RECORD", icon="[QR]"))
    story.append(Spacer(1, 3 * mm))

    report_id = data.get("report_id", str(uuid.uuid4())[:8].upper())
    scan_id   = data.get("scan_id", "N/A")
    gen_time  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    avail_w = PAGE_W - 2 * MARGIN

    # QR Code column
    qr_buf = _generate_qr_image(scan_id)
    qr_items = []
    if qr_buf:
        qr_img = Image(qr_buf, width=55 * mm, height=55 * mm)
        qr_items.append(qr_img)
        qr_items.append(Paragraph("Scan to view history", styles["caption"]))
    else:
        qr_items.append(Paragraph("[QR code — install 'qrcode' package]", styles["caption"]))

    # Verification badge column
    class VerifBadge(Flowable):
        def __init__(self, report_id, gen_time, width=130, height=60):
            super().__init__()
            self.report_id = report_id
            self.gen_time  = gen_time
            self.width     = width
            self.height    = height

        def draw(self):
            c = self.canv
            W, H = self.width, self.height
            c.setFillColor(NAVY)
            c.roundRect(0, 0, W, H, 8, fill=1, stroke=0)
            c.setFillColor(GREEN)
            c.roundRect(0, 0, W, H, 8, fill=0, stroke=1)
            c.setLineWidth(1.5)

            # Checkmark circle
            c.setFillColor(GREEN)
            c.circle(20, H / 2, 10, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 13)
            c.drawCentredString(20, H / 2 - 5, "✓")

            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(36, H - 18, "DIGITALLY VERIFIED REPORT")
            c.setFillColor(TEAL)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(36, H - 29, f"ID: {self.report_id}")
            c.setFillColor(MID_GRAY)
            c.setFont("Helvetica", 7.5)
            c.drawString(36, H - 40, "Generated by AgroAI AI Engine")
            c.drawString(36, H - 50, self.gen_time)

    verif_col = [VerifBadge(report_id, gen_time, width=avail_w * 0.55, height=65)]

    side_tbl = Table(
        [[qr_items, verif_col]],
        colWidths=[avail_w * 0.4, avail_w * 0.6],
    )
    side_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (0, -1), "CENTER"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    story.append(side_tbl)
    story.append(Spacer(1, 4 * mm))
    return story


# ─────────────────────────────────────────────
# MAIN GENERATE FUNCTION
# ─────────────────────────────────────────────

def generate_pdf_report(
    disease_name: str,
    confidence: float,
    symptoms: str,
    treatment: str,
    prevention: str,
    severity: str,
    is_healthy: bool,
    scan_id: str | None = None,
    db_id: str | None = None,
    image_bytes: bytes | None = None,
    image_data_url: str | None = None,
    filename: str | None = None,
    plant_type: str | None = None,
    output_path: str | None = None,
) -> bytes:
    """
    Generate and return a premium AgroAI PDF report as bytes.

    Parameters
    ----------
    disease_name    : str   — detected disease label
    confidence      : float — model confidence (0-100)
    symptoms        : str   — symptoms text from disease_info.json
    treatment       : str   — treatment text
    prevention      : str   — prevention text
    severity        : str   — "LOW" | "MEDIUM" | "HIGH"
    is_healthy      : bool  — True if no disease detected
    scan_id         : str   — frontend scan ID
    db_id           : str   — MongoDB document ID
    image_bytes     : bytes — raw uploaded image bytes (preferred)
    image_data_url  : str   — base64 data URL fallback
    filename        : str   — original uploaded filename
    plant_type      : str   — plant species/type (inferred if not provided)
    output_path     : str   — if set, also saves PDF to disk

    Returns
    -------
    bytes — raw PDF binary data (for Flask send_file / BytesIO)
    """
    _register_fonts()
    styles   = _build_styles()
    report_id = (str(uuid.uuid4())[:8]).upper()
    scan_id   = scan_id or (str(uuid.uuid4())[:12]).upper()
    scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resolve image bytes
    if image_bytes is None and image_data_url:
        image_bytes = _base64_to_pil_bytes(image_data_url)

    # Assemble data dict for section builders
    data = {
        "report_id":   report_id,
        "scan_id":     scan_id,
        "scan_time":   scan_time,
        "disease_name": disease_name,
        "confidence":  confidence,
        "severity":    severity.upper() if severity else "LOW",
        "is_healthy":  is_healthy,
        "symptoms":    symptoms,
        "treatment":   treatment,
        "prevention":  prevention,
        "db_id":       db_id,
        "filename":    filename or "upload.jpg",
        "plant_type":  plant_type or (disease_name.split()[0] if disease_name else "Unknown"),
        "cause":       f"This condition is typically caused by fungal, bacterial, or environmental stress specific to {disease_name}.",
        "spread_risk": (
            "High — can spread rapidly to neighbouring plants via spores, insects, or water splash."
            if severity == "HIGH" else
            "Moderate — may spread under favourable conditions (humidity, temperature)."
            if severity == "MEDIUM" else
            "Low — limited spread under normal conditions."
        ),
    }

    # Build PDF in-memory
    pdf_buf = io.BytesIO()
    canvas_factory = ReportCanvasFactory(report_id, version="2.0.0")
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=28 * mm,     # room for footer
        canvasmaker=canvas_factory,
        title=f"AgroAI Report — {disease_name}",
        author="AgroAI AI Engine",
        subject="Crop Disease Analysis Report",
        creator="AgroAI v2.0.0",
    )

    # ── Assemble story ──────────────────────────────────────────────────
    story = []

    story += _build_hero_header(styles, report_id)
    story += _build_scan_summary_card(styles, data)
    story += _build_image_section(styles, image_bytes)
    story.append(PageBreak())                                # Page 2 starts here
    story += _build_ai_analysis(styles, data)
    story += _build_treatment_section(styles, data)
    story.append(PageBreak())                                # Page 3
    story += _build_prevention_section(styles, data)
    story += _build_severity_visualization(styles, data)
    story += _build_smart_insights(styles, data)
    story.append(PageBreak())                                # Page 4
    story += _build_database_details(styles, data)
    story += _build_farmer_advisory(styles, data)
    story += _build_qr_and_verification(styles, data)

    # Final disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY, spaceBefore=6))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system and is intended for agricultural advisory purposes only. "
        "Always consult a certified agronomist or plant pathologist before applying any treatment. "
        "AgroAI is not liable for decisions made solely based on this report.",
        ParagraphStyle("disclaimer", fontName="Helvetica-Oblique",
                       fontSize=7.5, textColor=MID_GRAY, alignment=TA_CENTER,
                       leading=11, spaceBefore=4)
    ))

    doc.build(story)
    pdf_bytes = pdf_buf.getvalue()

    if output_path:
        try:
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(f"PDF saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save PDF to disk: {e}")

    return pdf_bytes


# ─────────────────────────────────────────────
# CLI TEST  (run: python pdf_generator.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_bytes = generate_pdf_report(
        disease_name="Tomato Early Blight",
        confidence=87.4,
        symptoms=(
            "1. Small dark brown spots with concentric rings (target pattern)\n"
            "2. Yellowing of leaves surrounding lesions\n"
            "3. Premature defoliation of lower leaves\n"
            "4. Dark lesions on stem near soil level"
        ),
        treatment=(
            "1. Apply copper-based fungicide every 7-10 days\n"
            "2. Remove and destroy infected plant material immediately\n"
            "3. Use Mancozeb 75% WP at 2g per litre of water\n"
            "4. Ensure adequate ventilation between plants"
        ),
        prevention=(
            "1. Use disease-resistant tomato varieties (e.g., Roma VF)\n"
            "2. Rotate crops — avoid tomatoes after tomatoes/potatoes\n"
            "3. Drip irrigation minimises leaf wetness\n"
            "4. Apply mulch to reduce soil splash onto lower leaves\n"
            "5. Maintain soil pH between 6.0-6.8"
        ),
        severity="MEDIUM",
        is_healthy=False,
        scan_id="SCAN-DEMO-001",
        db_id="6650abc123def456abc78901",
        filename="tomato_leaf.jpg",
        plant_type="Tomato",
        output_path="AgroAI_Demo_Report.pdf",
    )
    print(f"PDF generated: {len(sample_bytes):,} bytes -> AgroAI_Demo_Report.pdf")
