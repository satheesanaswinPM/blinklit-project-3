"""Generate a findings PDF from output/synthesis.json for deck creation."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SYN_PATH = ROOT / "output" / "synthesis.json"
OUT_PATH = ROOT / "docs" / "AI_Analysis_Engine_Findings.pdf"

GREEN = colors.HexColor("#0D9F6E")
GREEN_DARK = colors.HexColor("#05664A")
INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4B5563")
LIGHT = colors.HexColor("#ECF4F0")
WHITE = colors.white
AMBER = colors.HexColor("#B54708")


def _pretty(s: str) -> str:
    return (s or "").replace("_", " ")


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=GREEN_DARK,
            alignment=TA_CENTER,
            spaceAfter=8,
            leading=26,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=6,
            leading=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=GREEN_DARK,
            spaceBefore=14,
            spaceAfter=8,
            leading=18,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=4,
            leading=14,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=INK,
            leading=13,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            textColor=MUTED,
            leading=11,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=INK,
            leading=10,
        ),
        "cell_b": ParagraphStyle(
            "cell_b",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=INK,
            leading=10,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), GREEN_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), WHITE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(GREEN)
    canvas.rect(0, A4[1] - 4 * mm, A4[0], 4 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        18 * mm,
        10 * mm,
        "Category Discovery Engine  |  Findings for deck creation  |  Source: output/synthesis.json",
    )
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> Path:
    syn = json.loads(SYN_PATH.read_text(encoding="utf-8"))
    styles = build_styles()
    story = []

    corpus = syn.get("corpus") or {}
    signals = corpus.get("exploration_signals") or {}
    generated = str(syn.get("generated_at") or "")[:19].replace("T", " ")

    # Cover
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("AI Analysis Engine — Findings Report", styles["cover_title"]))
    story.append(Paragraph(syn.get("primary_question") or "", styles["cover_sub"]))
    story.append(
        Paragraph(
            f"Generated from synthesis.json · {generated} UTC<br/>"
            "Use these numbers as the source of truth for slides / Gamma / stakeholder decks.",
            styles["cover_sub"],
        )
    )
    story.append(Spacer(1, 10 * mm))

    kpi_data = [
        [
            Paragraph("<b>Corpus</b>", styles["cell"]),
            Paragraph("<b>Exploration-relevant</b>", styles["cell"]),
            Paragraph("<b>Relevance rate</b>", styles["cell"]),
            Paragraph("<b>Top barrier</b>", styles["cell"]),
        ],
        [
            Paragraph(f"{corpus.get('total_reviews', 0):,}", styles["cell_b"]),
            Paragraph(f"{corpus.get('exploration_relevant', 0):,}", styles["cell_b"]),
            Paragraph(f"{float(corpus.get('relevance_rate') or 0):.1%}", styles["cell_b"]),
            Paragraph("Quality distrust (211)", styles["cell_b"]),
        ],
    ]
    story.append(_table(kpi_data, [40 * mm, 45 * mm, 40 * mm, 50 * mm]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("<b>Executive summary</b>", styles["h2"]))
    story.append(Paragraph(syn.get("executive_summary") or "", styles["body"]))

    story.append(PageBreak())

    # 1. Corpus & signals
    story.append(Paragraph("1. Corpus & exploration signals", styles["h1"]))
    story.append(
        Paragraph(
            "Analysis corpus is the preprocessed set (aligned with sentiment / segments / exploration tags).",
            styles["small"],
        )
    )
    by_source = corpus.get("by_source") or {}
    src_rows = [[Paragraph("<b>Source</b>", styles["cell"]), Paragraph("<b>Rows</b>", styles["cell"])]]
    for k, v in by_source.items():
        src_rows.append([Paragraph(_pretty(str(k)), styles["cell"]), Paragraph(f"{int(v):,}", styles["cell"])])
    if len(src_rows) == 1:
        src_rows.append([Paragraph("play_store", styles["cell"]), Paragraph(f"{corpus.get('total_reviews', 0):,}", styles["cell"])])
    story.append(_table(src_rows, [90 * mm, 40 * mm]))
    story.append(Spacer(1, 4 * mm))

    sig_rows = [
        [
            Paragraph("<b>Exploration signal</b>", styles["cell"]),
            Paragraph("<b>Count</b>", styles["cell"]),
            Paragraph("<b>Share of corpus</b>", styles["cell"]),
        ]
    ]
    total = max(int(corpus.get("total_reviews") or 1), 1)
    order = [
        "want_to_explore_blocked",
        "stuck_in_routine",
        "explored_new",
        "unclear",
        "noise",
    ]
    for key in order:
        n = int(signals.get(key) or 0)
        sig_rows.append(
            [
                Paragraph(_pretty(key), styles["cell"]),
                Paragraph(f"{n:,}", styles["cell"]),
                Paragraph(f"{n / total:.1%}", styles["cell"]),
            ]
        )
    story.append(_table(sig_rows, [70 * mm, 35 * mm, 40 * mm]))

    # segments from notes
    story.append(Paragraph("User segments (KMeans on feedback)", styles["h2"]))
    seg_rows = [
        [
            Paragraph("<b>Segment</b>", styles["cell"]),
            Paragraph("<b>Count</b>", styles["cell"]),
            Paragraph("<b>Implication for deck</b>", styles["cell"]),
        ],
        [Paragraph("Price Sensitive", styles["cell"]), Paragraph("1,736", styles["cell"]), Paragraph("Fee/markup fear blocks trial", styles["cell"])],
        [Paragraph("Routine Buyers", styles["cell"]), Paragraph("554", styles["cell"]), Paragraph("Same milk/bread loop; never browse", styles["cell"])],
        [Paragraph("Explorers", styles["cell"]), Paragraph("348", styles["cell"]), Paragraph("Open to try if trust + clarity", styles["cell"])],
        [Paragraph("Impulse Buyers", styles["cell"]), Paragraph("180", styles["cell"]), Paragraph("Attach if surfaced in-session", styles["cell"])],
    ]
    story.append(_table(seg_rows, [45 * mm, 25 * mm, 105 * mm]))

    # 2. Barriers
    story.append(Paragraph("2. Barriers ranked (why users avoid new categories)", styles["h1"]))
    bar_rows = [
        [
            Paragraph("<b>Rank</b>", styles["cell"]),
            Paragraph("<b>Barrier</b>", styles["cell"]),
            Paragraph("<b>Mentions</b>", styles["cell"]),
            Paragraph("<b>Share</b>", styles["cell"]),
            Paragraph("<b>Severity</b>", styles["cell"]),
        ]
    ]
    for i, b in enumerate(syn.get("barriers_ranked") or [], start=1):
        bar_rows.append(
            [
                Paragraph(str(i), styles["cell"]),
                Paragraph(_pretty(str(b.get("barrier"))), styles["cell"]),
                Paragraph(str(b.get("mentions")), styles["cell"]),
                Paragraph(f"{float(b.get('share') or 0):.1%}", styles["cell"]),
                Paragraph(str(b.get("severity") or ""), styles["cell"]),
            ]
        )
    story.append(_table(bar_rows, [15 * mm, 75 * mm, 25 * mm, 25 * mm, 25 * mm]))

    story.append(PageBreak())

    # 3. Hypotheses
    story.append(Paragraph("3. Hypotheses × evidence", styles["h1"]))
    story.append(
        Paragraph(
            "Evidence mentions = barrier-linked counts from exploration tags (not identity/eval placeholders).",
            styles["small"],
        )
    )
    hyp_rows = [
        [
            Paragraph("<b>ID</b>", styles["cell"]),
            Paragraph("<b>Statement</b>", styles["cell"]),
            Paragraph("<b>Linked barrier</b>", styles["cell"]),
            Paragraph("<b>Evidence</b>", styles["cell"]),
        ]
    ]
    for h in syn.get("hypotheses") or []:
        hyp_rows.append(
            [
                Paragraph(str(h.get("id")), styles["cell_b"]),
                Paragraph(str(h.get("statement") or ""), styles["cell"]),
                Paragraph(_pretty(str(h.get("linked_barrier") or "")), styles["cell"]),
                Paragraph(str(h.get("evidence_mentions")), styles["cell_b"]),
            ]
        )
    story.append(_table(hyp_rows, [12 * mm, 100 * mm, 45 * mm, 18 * mm]))

    # 4. JTBD + unmet needs
    story.append(Paragraph("4. Jobs to be done", styles["h1"]))
    for i, job in enumerate(syn.get("jobs_to_be_done") or [], start=1):
        story.append(Paragraph(f"<b>JTBD {i}.</b> {job.get('job', '')}", styles["body"]))
        story.append(
            Paragraph(
                f"Situation: {job.get('situation', '')} · Desired: {job.get('desired_outcome', '')} · "
                f"Workaround: {job.get('current_workaround', '')}",
                styles["small"],
            )
        )

    story.append(Paragraph("5. Unmet needs", styles["h1"]))
    need_rows = [
        [
            Paragraph("<b>Priority</b>", styles["cell"]),
            Paragraph("<b>Need</b>", styles["cell"]),
            Paragraph("<b>Pain</b>", styles["cell"]),
            Paragraph("<b>Evidence cue</b>", styles["cell"]),
        ]
    ]
    for n in syn.get("unmet_needs") or []:
        need_rows.append(
            [
                Paragraph(str(n.get("priority") or ""), styles["cell_b"]),
                Paragraph(str(n.get("need") or ""), styles["cell"]),
                Paragraph(str(n.get("pain") or ""), styles["cell"]),
                Paragraph(str(n.get("evidence") or ""), styles["cell"]),
            ]
        )
    story.append(_table(need_rows, [18 * mm, 55 * mm, 55 * mm, 47 * mm]))

    story.append(PageBreak())

    # 6. Category opportunities
    story.append(Paragraph("6. Category opportunities (ranked)", styles["h1"]))
    cat_rows = [
        [
            Paragraph("<b>#</b>", styles["cell"]),
            Paragraph("<b>Category</b>", styles["cell"]),
            Paragraph("<b>Score</b>", styles["cell"]),
            Paragraph("<b>Blocked</b>", styles["cell"]),
            Paragraph("<b>Barrier to attack</b>", styles["cell"]),
            Paragraph("<b>Suggested experiment</b>", styles["cell"]),
            Paragraph("<b>Why now</b>", styles["cell"]),
        ]
    ]
    for c in syn.get("category_opportunities") or []:
        cat_rows.append(
            [
                Paragraph(str(c.get("rank")), styles["cell"]),
                Paragraph(str(c.get("category")), styles["cell_b"]),
                Paragraph(str(c.get("opportunity_score")), styles["cell"]),
                Paragraph(str(c.get("blocked_mentions")), styles["cell"]),
                Paragraph(_pretty(str(c.get("primary_barrier_to_attack") or "")), styles["cell"]),
                Paragraph(str(c.get("suggested_experiment") or ""), styles["cell"]),
                Paragraph(str(c.get("why_now") or ""), styles["cell"]),
            ]
        )
    story.append(_table(cat_rows, [10 * mm, 22 * mm, 16 * mm, 18 * mm, 38 * mm, 35 * mm, 36 * mm]))

    # 7. Experiments
    story.append(Paragraph("7. Testable experiments (MVP links)", styles["h1"]))
    for e in syn.get("testable_experiments") or []:
        story.append(
            Paragraph(
                f"<b>{e.get('id')}</b> — {e.get('name')} · Hypothesis {e.get('hypothesis_link')} · "
                f"Barrier: {_pretty(str(e.get('barrier') or ''))}",
                styles["body"],
            )
        )
        story.append(Paragraph(f"Intervention: {e.get('intervention', '')}", styles["small"]))
        story.append(
            Paragraph(
                f"Primary metric: {e.get('primary_metric', '')} · Guardrail: {e.get('guardrail', '')} · "
                f"{e.get('sample_size_note', '')}",
                styles["small"],
            )
        )

    # 8. Deck-ready metrics + features
    story.append(Paragraph("8. Deck-ready metrics & finalized product features", styles["h1"]))
    story.append(Paragraph("<b>North star</b> — % of MACs who purchase from ≥1 new product category each month", styles["body"]))
    story.append(
        Paragraph(
            "<b>Leading (Snacks rail)</b> — Attach rate of non-grocery SKU in same session · rail CTR · PDP→ATC<br/>"
            "<b>Leading (Home guarantee)</b> — First-time Home conversion among grocery-only cohorts<br/>"
            "<b>Lagging</b> — Category mix per MAC · repeat in new category · expanded basket / AOV · retention / CLV contribution<br/>"
            "<b>Guardrails</b> — Checkout completion drop ≤2% (rail) · return-rate spike &lt;3pp (guarantee) · no surge in hidden-fees tickets",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>Finalized MVP features</b>", styles["h2"]))
    story.append(
        Paragraph(
            "1) Contextual snacks discovery rail after grocery add (exp_discover_rail)<br/>"
            "2) Home first-buy quality guarantee + easy return (exp_first_buy_guarantee)<br/>"
            "3) Trust cues: ratings/comments, all-in price, quality/freshness badge<br/>"
            "4) Research console: Findings, Opportunities, Validation Desk, Prototype Lab",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>Recommendation logic (MVP)</b>", styles["h2"]))
    story.append(
        Paragraph(
            "Session context (grocery reorder) → category adjacency (snacks first) → barrier-aware treatment "
            "(trust/guarantee for high-distrust categories) → inventory availability → price transparency → "
            "social proof. Purchase/search/browse personalization, weather, festivals = roadmap, not MVP-critical.",
            styles["body"],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "<b>Source of truth:</b> output/synthesis.json · output/exploration_tags.csv · "
            "output/sentiment.csv · output/user_segments.csv · docs/architecture.md",
            styles["small"],
        )
    )
    story.append(
        Paragraph(
            "Regenerate this PDF: python scripts/export_findings_pdf.py",
            styles["small"],
        )
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="AI Analysis Engine Findings",
        author="Category Discovery Engine",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
