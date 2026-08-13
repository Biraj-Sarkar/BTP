"""Generate the project progress report as a PDF.

    python scripts/make_progress_report.py --out ~/Desktop/BTP/Anubhav/Progress_Report.pdf
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#8C2F39")
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5A5A5A")
RULE = colors.HexColor("#D8D8D8")
BAND = colors.HexColor("#F4F0F0")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=21, leading=25,
                                textColor=INK, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=11.5,
                                   leading=15, textColor=MUTED, spaceAfter=2),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontSize=8.5,
                               textColor=MUTED, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=13.5, leading=17,
                             textColor=ACCENT, spaceBefore=15, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=10.5, leading=13,
                             textColor=INK, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13.6,
                               textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.5, leading=13.4,
                                 textColor=INK, leftIndent=11, bulletIndent=2, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=8.2, leading=10.6,
                               textColor=INK),
        "cellh": ParagraphStyle("cellh", parent=base["Normal"], fontSize=8.2, leading=10.6,
                                textColor=colors.white, fontName="Helvetica-Bold"),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=8,
                                  leading=10.5, textColor=MUTED, spaceBefore=3),
        "callout": ParagraphStyle("callout", parent=base["Normal"], fontSize=9.3,
                                  leading=13, textColor=INK, alignment=TA_JUSTIFY),
    }


S = styles()


def para(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{item}", S["bullet"]) for item in items]


def table(rows, widths, header=True):
    data = []
    for index, row in enumerate(rows):
        style = "cellh" if (header and index == 0) else "cell"
        data.append([Paragraph(str(c), S[style]) for c in row])

    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF8F8")]),
        ]
    t.setStyle(TableStyle(commands))
    return t


def callout(text, tint=BAND):
    t = Table([[Paragraph(text, S["callout"])]], colWidths=[168 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
    ]))
    return t


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(21 * mm, 16 * mm, 189 * mm, 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(21 * mm, 11.5 * mm, "BTP — Non-Invasive Anemia Detection · Progress Report")
    canvas.drawRightString(189 * mm, 11.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(out_path: Path, figure: Path | None, real_figure: Path | None) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=21 * mm, rightMargin=21 * mm,
        topMargin=18 * mm, bottomMargin=22 * mm,
        title="BTP Progress Report — Non-Invasive Anemia Detection",
        author="Anubhav Jindal",
    )

    story = []
    story.append(para("Non-Invasive Anemia Detection", "title"))
    story.append(para("Estimating haemoglobin from photographs of the palpebral conjunctiva", "subtitle"))
    story.append(para(f"Progress report &middot; {date.today():%d %B %Y}", "meta"))

    # ---------------------------------------------------------------- summary
    story.append(para("1. Summary", "h1"))
    story.append(para(
        "This period was spent building the image-processing and learning pipeline end to end: data "
        "loading and clinical labelling, conjunctiva extraction, quality control, the regression "
        "scaffold, an evaluation methodology, and a serving API. The guiding constraints are an "
        "eventually deployed screening app and small clinical cohorts &mdash; both punish silent "
        "failure, so the pipeline is engineered to produce numbers that can be defended: patient-grouped "
        "cross-validation, out-of-fold reporting only, and every result compared against a "
        "predict-the-mean baseline."
    ))
    story.append(para(
        "A real labelled cohort was obtained: 52 photographs of 26 children, both eyes, with laboratory "
        "haemoglobin and a full blood count. Screening it produced two findings. The pipeline processes "
        "real photographs without error and 94% pass quality control, but a plain colour-threshold "
        "extraction &mdash; which scores almost perfectly on synthetic images &mdash; proved "
        "<b>unreliable on real tissue</b>; a refined extractor was then built against these captures and "
        "now produces clean masks across all 52. Separately, the per-patient WHO threshold was validated "
        "against real laboratory values: a fixed single cutoff would have missed 15% of the anemic "
        "children in this cohort. Both are reported in Section 4."
    ))
    story.append(para(
        "The regression model has <b>not</b> yet been trained on real patient photographs, so no "
        "accuracy figure in this report is a clinical result."
    ))

    # ------------------------------------------------------------- rebuild
    story.append(para("2. Pipeline design", "h1"))
    story.append(para(
        "The pipeline is a package of focused modules. The decision that matters most architecturally is "
        "that the inference path shares its preprocessing code with training but imports none of the "
        "training stack &mdash; so the deployed server cannot drift from the conditions the model was "
        "trained under, and does not load training machinery to answer a request."
    ))
    story.append(table([
        ["Area", "Design"],
        ["Extraction", "Built around a redness prior in CIELAB refined by a seeded grabCut; all backends "
                       "share one interface so a trained model swaps in without touching other stages"],
        ["Quality control", "Enforced, not merely recorded: failing captures are dropped from training; "
                            "the API returns a retake prompt rather than a number"],
        ["Clinical labelling", "WHO thresholds applied per patient from age and sex"],
        ["Evaluation", "Patient-grouped, haemoglobin-stratified cross-validation; out-of-fold predictions "
                       "only; always reported against a predict-the-mean baseline"],
        ["Checkpoints", "Self-describing bundle carrying weights, standardisation constants, preprocessing "
                        "settings and metrics"],
        ["Performance", "Preprocessing cached to disk, so segmentation runs once rather than once per epoch"],
        ["Data handling", "Adapters for both public datasets behind a common record type; adding the "
                          "hospital data requires one adapter"],
        ["Deployment", "FastAPI service loading one model at startup"],
        ["Testing", "19 automated tests, plus synthetic eye phantoms allowing development without data"],
    ], [34 * mm, 134 * mm]))

    # ------------------------------------------------------------- figure
    if figure and figure.exists():
        story.append(para("3. Choosing the extraction target: a controlled comparison", "h1"))
        story.append(para(
            "Two candidate colour rules were scored against known ground truth on synthetic phantoms. A "
            "brightness-based mask (bright, low-chroma pixels) scores a Dice coefficient of <b>0.000</b> "
            "&mdash; it selects the sclera and has no overlap with the conjunctiva. The redness-based "
            "extractor scores <b>0.999</b>. This is what fixed redness, not brightness, as the extraction "
            "target."
        ))
        story.append(Spacer(1, 3))
        story.append(Image(str(figure), width=168 * mm, height=56 * mm))
        story.append(para(
            "Left: input capture. Centre: region identified by the redness-based extractor, tinted "
            "green. Right: the square, undistorted crop passed to the regression network.", "caption"))
        story.append(Spacer(1, 5))
        story.append(callout(
            "<b>These are synthetic phantoms with flat colours and artificially clean separation.</b> "
            "A near-perfect score is expected and carries no clinical meaning. "
            "What the comparison establishes is that brightness-based masking targets the wrong tissue "
            "and redness-based extraction targets the right one. Section 4 shows what happens on real "
            "photographs."))

    # ------------------------------------------------------- real images
    story.append(para("4. Evaluation on real photographs", "h1"))
    story.append(para(
        "A local cohort of 26 children was obtained: both eyes photographed, giving 52 close-up captures "
        "of the everted lower eyelid, accompanied by laboratory haemoglobin, date of birth, gender, and a "
        "full blood count. The captures are well framed and in focus &mdash; tighter on the conjunctiva "
        "than typical field photography."
    ))
    story.append(para("4.1 Cohort", "h2"))
    story.append(table([
        ["Property", "Value"],
        ["Patients / images", "26 / 52 (both eyes, exactly two captures each)"],
        ["Haemoglobin", "8.1 &ndash; 14.3 g/dL (mean 11.2, sd 1.37)"],
        ["Anemic by WHO threshold", "13 of 26 (50%)"],
        ["Severe (Hb &lt; 9.0)", "3 patients"],
        ["Age", "3.4 &ndash; 16.7 years"],
        ["Additional labels", "Height, weight, socio-economic status, HCT, RBC, MCV, MCH, MCHC, RDW, platelets, MPV, TLC"],
    ], [45 * mm, 123 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "Two properties make this cohort more useful than its size suggests. The 50/50 anemic split is "
        "unusually balanced &mdash; most cohorts skew heavily normal, which is what makes a "
        "predict-the-mean baseline hard to beat. And because both eyes of one child are photographed, the "
        "two captures are highly correlated: the loader assigns them a shared patient identifier so the "
        "grouped splitter cannot separate them; under a naive image-level split this dataset would "
        "produce substantially inflated results."
    ))
    story.append(para("4.2 The pipeline runs on real data", "h2"))
    story.append(table([
        ["Measure", "Result"],
        ["Images processed without error", "52 of 52"],
        ["Passed quality control", "49 of 52 (94%)"],
        ["Focus (Laplacian variance)", "median 85.3, range 25.4 &ndash; 294.2"],
        ["Mask coverage of frame", "median 0.211, range 0.041 &ndash; 0.449"],
        ["Rejections", "3, all blur (25.4, 34.2, 34.9 against a threshold of 35.0)"],
    ], [58 * mm, 110 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "Two of the three rejections sit marginally below the threshold on captures that look usable to "
        "the eye. The blur threshold was chosen on synthetic images and appears slightly too aggressive "
        "for real photographs; it should be recalibrated once ground-truth masks make the downstream "
        "effect measurable."
    ))

    story.append(para("4.3 Extraction: failure found, then fixed", "h2"))
    story.append(para(
        "Quality-control statistics say whether a capture is usable, not whether the segmentation found "
        "the right structure. Visual inspection of the overlays answers that, and the answer is "
        "unfavourable: on a substantial fraction of the 52 real captures the colour prior bleeds onto the "
        "eyelashes and lower-lid skin, and occasionally onto the sclera. On several it extends well below "
        "the lash line onto the cheek."
    ))
    story.append(para(
        "The cause is straightforward: eyelashes and peri-orbital skin are reddish-brown enough to pass "
        "a redness test, and neither is dark enough in absolute terms to be removed by a lightness gate."
    ))
    story.append(para(
        "A refined extractor was then built against these captures. It seeds a mask-initialised grabCut "
        "with what actually distinguishes the tissue: conjunctiva is smooth where the lash zone is "
        "high-frequency (a local-texture gate), sclera is bright and desaturated, iris and pupil are "
        "dark, and specular highlights sit on the wet tissue itself and are kept rather than discarded. "
        "Re-screened over all 52 captures, the refined extractor produces clean, tissue-hugging masks on "
        "visual review, and the median mask area halves &mdash; consistent with the bleed being removed "
        "rather than the tissue. The residual failure mode is a barely-everted lid, which is a capture "
        "problem rather than a segmentation one. The figure below shows the plain threshold (top) "
        "against the refined extractor (bottom) on the same captures."
    ))
    if real_figure and real_figure.exists():
        story.append(Spacer(1, 3))
        story.append(Image(str(real_figure), width=168 * mm, height=84 * mm))
        story.append(para(
            "Four real captures. Top row: the plain colour threshold, which bleeds onto lashes and "
            "skin. Bottom row: the refined seeded-grabCut extractor on the same images.", "caption"))
    story.append(Spacer(1, 5))
    story.append(callout(
        "<b>A method scoring 0.999 on synthetic images was visibly unreliable on real tissue</b> &mdash; "
        "exactly the gap synthetic validation cannot reveal. The failure was characterised, a refined "
        "extractor was built against the real captures, and visual review across all 52 now shows clean "
        "masks. What remains missing is a <i>measured</i> Dice on real photographs, which requires the "
        "Eyes-defy-anemia ground-truth masks; until then the learned segmenter remains the primary "
        "long-term path."))

    story.append(para("4.4 The per-patient threshold validated on real data", "h2"))
    story.append(para(
        "A single fixed anemia cutoff (commonly 11.0 g/dL) is a tempting simplification, and this cohort "
        "measures exactly what it would cost, because its ages span three WHO threshold bands: 11.0 "
        "below five years, 11.5 from five to eleven, and 12.0 from twelve to fourteen. Twenty of the "
        "twenty-six children fall in the middle band."
    ))
    story.append(table([
        ["Patient", "Hb (g/dL)", "Age", "WHO threshold", "Correct", "Fixed 11.0 rule"],
        ["10", "11.0", "11.1", "11.5", "anemic", "<b>normal</b>"],
        ["14", "11.4", "6.6", "11.5", "anemic", "<b>normal</b>"],
        ["16", "11.3", "8.4", "11.5", "anemic", "<b>normal</b>"],
        ["21", "11.4", "7.2", "11.5", "anemic", "<b>normal</b>"],
    ], [20 * mm, 22 * mm, 16 * mm, 30 * mm, 30 * mm, 50 * mm]))
    story.append(Spacer(1, 4))
    story.append(callout(
        "<b>A fixed 11.0 rule misclassifies 4 of 26 patients (15%), and every one is a false negative "
        "&mdash; an anemic child labelled normal.</b> The errors cluster just below 11.5 because most of "
        "this cohort sits in the five-to-eleven band, where the correct cutoff is half a gram above the "
        "fixed value. This error occurs before the model makes a single prediction, so no amount of "
        "regression accuracy could recover it &mdash; which is why the pipeline predicts a continuous Hb "
        "and applies the threshold per patient."))

    # ------------------------------------------------------------- status
    story.append(para("5. Verification status", "h1"))
    story.append(para("5.1 Verified", "h2"))
    story.extend(bullets([
        "All pipeline stages execute end to end; 19 of 19 automated tests pass.",
        "Cross-validated training completes on synthetic data: MAE 0.664 &plusmn; 0.096 g/dL, R&sup2; 0.880, "
        "1.45 g/dL better than a predict-the-mean baseline.",
        "A saved checkpoint carries everything needed to serve it, loads, and produces a haemoglobin "
        "estimate.",
        "Quality control correctly rejects blurred captures and returns a retake prompt.",
        "Clinical thresholds resolve correctly across age and sex.",
        "The pipeline processes real photographs end to end: 52 of 52 without error, 94% passing "
        "quality control.",
        "The per-patient WHO threshold is validated against real laboratory values: a fixed single "
        "cutoff misclassifies 15% of this cohort, all as false negatives (Section 4.4).",
    ]))
    story.append(para("5.2 Measured and found wanting", "h2"))
    story.extend(bullets([
        "The plain colour threshold is unreliable on real tissue, bleeding onto eyelashes and lid "
        "skin. A refined seeded-grabCut extractor built against the same captures now produces clean "
        "masks on visual review across all 52 (Section 4.3); a measured Dice still awaits ground-truth "
        "masks.",
        "The blur threshold looks slightly too aggressive for real captures: two of three rejections "
        "were marginal on images that appear usable.",
    ]))
    story.append(para("5.3 Not yet established", "h2"))
    story.extend(bullets([
        "Any <i>accuracy</i> on real photographs. The local cohort is labelled and could in principle "
        "supply one, but 26 patients is far too few for a defensible confidence interval, and "
        "segmentation on these images is known to be unreliable. Training on it is worthwhile only after "
        "the segmenter works.",
        "The segmentation model has not been trained — this requires the ground-truth masks.",
        "The CP-AnemiC loader has not been checked against a real download.",
        "No clinical validation of any kind has been performed.",
    ]))

    # ------------------------------------------------------------- data
    story.append(para("6. Data", "h1"))
    story.append(para(
        "Two public datasets were identified and assessed, and a small labelled cohort is held locally. "
        "The professor has arranged hospital data, which will be the primary source; these sources "
        "allow development and benchmarking in the meantime."
    ))
    story.append(table([
        ["Source", "Images", "Hb", "Masks", "Role"],
        ["Eyes-defy-anemia", "218", "Yes", "Yes",
         "Only source of pixel-level ground truth; trains the segmenter"],
        ["CP-AnemiC", "710", "Yes", "No", "Additional volume for the regressor"],
        ["Local cohort", "52", "Yes", "No",
         "26 children, both eyes, with full blood count. Balanced 50/50 anemic. Too small to train on alone"],
    ], [32 * mm, 15 * mm, 10 * mm, 15 * mm, 96 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "Two consequences follow. First, roughly 900 images is a small corpus: it constrains model size, "
        "makes cross-validation necessary rather than optional, and means confidence intervals belong on "
        "every reported number. Second, the two sets differ in age, ethnicity, camera and lighting, so a "
        "model trained on one and evaluated on the other measures domain transfer as much as pallor. "
        "Results should be reported per source as well as pooled."
    ))
    story.append(Spacer(1, 3))
    story.append(para(
        "The local cohort carries a full blood count alongside haemoglobin. RDW and MCV distinguish "
        "iron-deficiency anemia from other types, which raises a possible extension: predicting which "
        "anemia rather than only whether. That is beyond current scope but worth recording. Provenance "
        "and licensing are unconfirmed, which matters if any figure derived from these images appears in "
        "a published write-up."
    ))

    # ------------------------------------------------------------- risks
    story.append(para("7. Risks", "h1"))
    story.append(table([
        ["Risk", "Mitigation"],
        ["Small dataset produces a model that does not beat the mean baseline",
         "Baseline comparison reported by construction, so this cannot go unnoticed"],
        ["Colour heuristic proves inadequate on real photographs",
         "Expected; the Mask2Former segmenter is the primary path and the heuristic only a fallback"],
        ["Hospital data arrives in an unanticipated format",
         "Loaders are isolated behind a common record type; an inspection command verifies parsing before training"],
        ["Results overstated in the write-up",
         "Grouped splits, out-of-fold reporting and baseline comparison are built into the pipeline rather "
         "than left to discipline"],
    ], [62 * mm, 106 * mm]))

    # ------------------------------------------------------------- next
    story.append(para("8. Next steps", "h1"))
    story.append(table([
        ["#", "Step", "Outcome"],
        ["1", "Download both datasets and verify parsed counts against published figures",
         "Confidence the loaders are correct"],
        ["2", "Train the segmentation model on Eyes-defy-anemia",
         "First trained component, and the first measured Dice for the refined extractor"],
        ["3", "Benchmark all segmentation backends against ground truth",
         "<b>First real measurement in the project</b>; quantifies the rework"],
        ["4", "Train the regressor; report per-source and pooled with confidence intervals",
         "First meaningful accuracy estimate"],
        ["5", "Recalibrate the blur threshold against real captures", "Fewer spurious retake prompts"],
        ["6", "Pool the local cohort with the public datasets once segmentation is reliable",
         "26 balanced, well-labelled patients added to training"],
        ["7", "Build the application around the existing API", "Working demonstrator"],
    ], [8 * mm, 78 * mm, 82 * mm]))

    story.append(Spacer(1, 8))
    story.append(callout(
        "<b>Assessment.</b> The pipeline is in a state where real data can be introduced and produce "
        "trustworthy numbers, and it has been exercised on real photographs rather than synthetic ones. "
        "Extraction — this period's focus — was taken from a failing colour threshold to clean masks "
        "across the full real cohort, with the remaining step being a measured Dice against ground-truth "
        "masks. The scientific question, whether conjunctival pallor predicts haemoglobin accurately "
        "enough in this cohort to be useful, remains open and is what the next phase must answer."))

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--real-figure", type=Path, default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.figure, args.real_figure)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
