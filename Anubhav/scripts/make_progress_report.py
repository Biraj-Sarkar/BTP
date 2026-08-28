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


def build(out_path: Path, figure: Path | None, real_figure: Path | None,
          diagnostics_figure: Path | None = None) -> None:
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
        "The principal work of this period was a <b>head-to-head comparison of the two candidate "
        "pipelines</b>, since one must eventually be deployed. Both were trained and scored on identical "
        "patients under an identical protocol, across five feature representations and the full metric "
        "suite (Section 6). Pipeline A leads B on 10 of 11 metrics fitted and 9 of 11 held out, and on "
        "every parameter-matched framing, but a paired test shows the two are <b>not statistically "
        "separable at 26 patients</b> (p = 0.408). The <b>deployed model</b> \u2014 pipeline A\u2019s "
        "erythema representation with the quality gate applied before fitting \u2014 reaches in-sample "
        "MAE 0.918 g/dL and R\u00b2 +0.195, and on unseen patients MAE 1.033 and R\u00b2 <b>+0.007</b> "
        "against a leave-one-out baseline of 1.083 and \u22120.082: the first configuration here to beat "
        "its baseline on patients it has not seen, though a permutation test still puts it short of "
        "significance (p = 0.067). Its coefficients also pass a physiological check that the earlier "
        "channel-mean model failed (Section 6.9). The regression network has not been trained on real "
        "photographs, so <b>no accuracy figure in this report is a clinical result</b>."
    ))

    story.append(para("1.1 Abbreviations", "h2"))
    story.append(para(
        "Defined here on first use; the full glossary is in <font face='Courier'>GLOSSARY.md</font>."
    ))
    story.append(table([
        ["Term", "Expansion", "Term", "Expansion"],
        ["Hb", "Haemoglobin (g/dL)", "MAE", "Mean Absolute Error"],
        ["WHO", "World Health Organization", "MSE / RMSE", "Mean Squared Error / its root"],
        ["QC", "Quality Control", "R\u00b2", "Coefficient of determination"],
        ["ROI", "Region Of Interest", "LoA", "Limits of Agreement (Bland\u2013Altman)"],
        ["CIELAB", "CIE L\u002aa\u002ab\u002a colour space", "TP / FP / TN / FN", "True/False Positive/Negative"],
        ["L\u002a / a\u002a / b\u002a", "Lightness / green\u2013red / blue\u2013yellow axes", "PPV / NPV", "Positive / Negative Predictive Value"],
        ["CLAHE", "Contrast Limited Adaptive Histogram Equalisation", "F1", "Harmonic mean of precision and recall"],
        ["RGB / sRGB / P3", "Colour encodings and profiles", "Dice / IoU", "Segmentation overlap measures"],
        ["CNN", "Convolutional Neural Network", "CV / LOO", "Cross-Validation / Leave-One-Out"],
        ["ResNet", "Residual Network (CNN architecture)", "OLS", "Ordinary Least Squares"],
        ["CBC", "Complete Blood Count", "L1 / L2", "Absolute / squared coefficient penalties"],
    ], [24 * mm, 60 * mm, 24 * mm, 60 * mm]))

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

    # ------------------------------------------------- colour + lighting
    story.append(para("5. Colour normalisation and lighting robustness", "h1"))
    story.append(para(
        "Lighting corrupts a photograph in two independent ways: a colour cast and a brightness shift. "
        "Two strategies address them. Gray-world white balance rescales the RGB channels until their "
        "means agree, correcting the <i>cast</i>. Applying contrast-limited histogram equalisation to the "
        "CIELAB lightness channel corrects <i>brightness</i> while leaving colour untouched; feeding the "
        "network only the a\u002a/b\u002a channels then makes it blind to brightness entirely."
    ))
    story.append(para("5.1 Measured against laboratory haemoglobin", "h2"))
    story.append(para(
        "Both were scored on the local cohort using identical masks, so only the colour treatment "
        "differed. A structural point governs the comparison: equalising the lightness channel leaves "
        "a\u002a/b\u002a mathematically unchanged, so for the values the regressor consumes, that path is "
        "equivalent to no colour correction at all. The real contest is corrected versus uncorrected "
        "colour."
    ))
    story.append(table([
        ["Measure", "Uncorrected colour", "Gray-world corrected"],
        ["Correlation with laboratory Hb (Spearman)", "+0.02", "<b>+0.30</b>"],
        ["Left-vs-right eye disagreement (median a\u002a)", "2.69", "<b>2.02</b>"],
    ], [78 * mm, 45 * mm, 45 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "Without colour correction the redness-haemoglobin relationship is essentially absent; per-shot "
        "illuminant differences swamp it. Correction recovers a modest but real association, and the same "
        "child\u2019s two eyes agree about 25% better."
    ))

    story.append(para("5.2 Advantages and limitations of each approach", "h2"))
    story.append(table([
        ["Property", "Gray-world white balance", "CIELAB (CLAHE on L\u002a, a\u002a/b\u002a input)"],
        ["Corrects illuminant <b>colour</b>", "<b>Yes</b> \u2014 the cast is equalised", "No \u2014 the cast passes into a\u002a/b\u002a untouched"],
        ["Corrects illuminant <b>brightness</b>", "No", "<b>Yes</b> \u2014 equalised, then discarded entirely"],
        ["Preserves captured colour exactly", "No \u2014 the shift depends on frame composition", "<b>Yes</b> \u2014 a\u002a/b\u002a are mathematically unchanged"],
        ["Invariance guarantee", "None \u2014 a heuristic correction", "<b>Provable</b> \u2014 brightness is absent from the input"],
        ["Signal retained", "<b>Full RGB</b>, including the pale-is-lighter cue", "Colour only \u2014 the lightness cue is discarded"],
        ["Robustness across devices", "<b>Better</b> for colour; casts equalised", "Better for exposure; weaker for colour"],
        ["Failure mode", "Non-grey scenes skew the correction; a near-empty channel is amplified", "Warm or cool lighting is read as physiology"],
        ["Measured here", "<b>Spearman +0.30; eye gap 2.02; Hb SD 1.63</b>", "+0.02; 2.69; 2.74"],
    ], [40 * mm, 64 * mm, 64 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "The two correct <b>orthogonal</b> axes of the same problem, so they are complementary rather "
        "than competing. That motivated testing a hybrid \u2014 gray-world first, then a\u002a/b\u002a input \u2014 "
        "which is reported in Section 6.3. On present evidence gray-world alone is preferred, because "
        "discarding the lightness channel costs more information than the brightness invariance returns "
        "at this data scale."
    ))

    story.append(para("5.3 Lighting stress test", "h2"))
    story.append(para(
        "One volunteer\u2019s eye was photographed under six illuminants (white, red, green, blue, magenta, "
        "purple). Because it is the same eye minutes apart, the true haemoglobin is constant and every "
        "reported difference is measurement error \u2014 no blood test is required for the test to be "
        "informative."
    ))
    story.append(table([
        ["Spread across six illuminants (lower is better)", "Gray-world", "CIELAB"],
        ["Redness (a\u002a) standard deviation", "<b>17.58</b>", "32.76"],
        ["Haemoglobin estimate standard deviation (g/dL)", "<b>1.63</b>", "2.74"],
    ], [78 * mm, 45 * mm, 45 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "Both figures are far too wide to be usable \u2014 a 4.5 g/dL swing from lighting alone spans the "
        "clinical range. Inspection of the masks explains why: <b>both pipelines segment correctly only "
        "under neutral light</b>, so the spread measures extraction collapsing as much as colour drift."
    ))
    story.append(Spacer(1, 3))
    story.append(table([
        ["What this test design does well", "What it cannot establish"],
        ["The subject is constant, so the true haemoglobin is identical in all six frames and every "
         "reported difference is measurement error \u2014 no laboratory value is needed",
         "One subject and one eye, so it characterises failure modes rather than accuracy"],
        ["Isolates illumination as the only varying factor",
         "The illuminants are far harsher than any clinical setting, so it maps where the method "
         "breaks, not whether it breaks in practice"],
        ["Exposes a failure that neither synthetic images nor the cohort revealed",
         "Segmentation collapse and colour drift are confounded, so the spread ratios are directional "
         "rather than precise"],
    ], [84 * mm, 84 * mm]))
    story.append(Spacer(1, 4))
    story.append(callout(
        "<b>The underlying cause is physical, not algorithmic.</b> In four of the six illuminants a colour "
        "channel is effectively dead \u2014 under blue light the red channel averages 0.9 of 255, under red "
        "and purple the green channel averages about 1. The lamp emitted nothing there, so the sensor "
        "recorded nothing there, and no normalisation can recover a measurement that was never made. "
        "Gray-world then divides by that near-zero mean and amplifies sensor noise. The practical remedy "
        "is a quality-control check on illuminant neutrality: the ratio of brightest to dimmest channel "
        "mean separates acceptable captures (1.7) from unusable ones (22\u2013200) by two orders of "
        "magnitude."))

    # ------------------------------------------------- quantitative eval
    story.append(para("6. Quantitative evaluation", "h1"))
    story.append(para(
        "How the haemoglobin figure is produced, and how well it performs. No library converts an eye "
        "photograph into a haemoglobin value; the libraries used supply a network architecture and "
        "weights pretrained on general photographs. The mapping from image to haemoglobin exists only "
        "because it is trained on paired examples, and it is the substance of the project."
    ))
    story.append(para(
        "Mechanically: the 224\u00d7224 crop is normalised to the statistics the pretrained weights expect, "
        "passed through a ResNet-18 whose classification head is replaced by a single-output linear "
        "layer, and the resulting scalar is converted to g/dL using the training mean and standard "
        "deviation stored inside the checkpoint. That value is then compared with the patient\u2019s WHO "
        "threshold."
    ))
    story.append(para("6.1 Metrics reported", "h2"))
    story.append(table([
        ["Metric", "Why it is reported"],
        ["MSE / RMSE / MAE", "Squared error is the training objective; MAE is the figure a clinician reads as typical error"],
        ["Median absolute error", "Robust to a single catastrophic case, so divergence from MAE reveals outliers"],
        ["Bias", "Systematic offset; positive bias means over-estimating Hb, i.e. missing anaemia"],
        ["R\u00b2", "Variance explained; zero means no better than predicting the mean, negative means worse"],
        ["Bland\u2013Altman limits of agreement", "The standard clinical comparison of a device against a laboratory reference"],
        ["Recall (sensitivity)", "Fraction of anaemic patients caught \u2014 the error that sends an unwell patient home"],
        ["Precision, specificity, NPV", "Cost of false alarms, and trustworthiness of a negative result"],
        ["F1, balanced accuracy", "Summaries that resist the class imbalance which inflates plain accuracy"],
    ], [42 * mm, 126 * mm]))

    story.append(para("6.2 Synthetic phantoms \u2014 the machinery works", "h2"))
    story.append(table([
        ["Metric", "Model", "Baseline"],
        ["MAE (g/dL)", "<b>0.666</b>", "2.119"],
        ["RMSE (g/dL)", "0.836", "2.499"],
        ["R\u00b2", "<b>0.888</b>", "0.000"],
        ["Recall / Precision / F1", "<b>1.000</b> / 0.973 / 0.986", "0.000 / \u2014 / 0.000"],
    ], [56 * mm, 56 * mm, 56 * mm]))
    story.append(Spacer(1, 3))
    story.append(para(
        "Reported for completeness and as verification of the metric code. The phantoms are flat-coloured "
        "shapes whose redness is a deterministic function of haemoglobin, so a strong score is expected "
        "by construction and carries no clinical meaning."
    ))

    story.append(para("6.3 Real cohort \u2014 the linear floor is not yet met", "h2"))
    story.append(para(
        "No network has been trained on real data. The models below are deliberately <b>linear</b> \u2014 "
        "ordinary least squares and ridge regression over colour statistics \u2014 evaluated "
        "leave-one-patient-out. Their purpose is to establish the floor a neural network must clear: if a "
        "simple model already explains nothing, that bounds how much signal lies in summary colour and "
        "defines what the network has to find instead."
    ))
    story.append(para(
        "Five feature representations were tested for each pipeline, including two designed to be "
        "invariant to illumination intensity: chromaticity coordinates, R/(R+G+B), and the erythema "
        "index, log(R/G) \u2014 the dermatology standard for measuring redness. Leave-one-patient-out:"
    ))
    story.append(table([
        ["Representation", "k", "A: MAE", "A: R\u00b2", "A: F1", "B: MAE", "B: R\u00b2", "B: F1"],
        ["means (R, G, B)", "3", "1.073", "\u22120.166", "0.688", "1.163", "\u22120.260", "0.562"],
        ["chromaticity", "2", "1.063", "\u22120.103", "0.688", "1.144", "\u22120.199", "0.647"],
        ["<b>erythema index</b>", "2", "<b>1.062</b>", "<b>\u22120.094</b>", "<b>0.727</b>", "1.143", "\u22120.192", "0.647"],
        ["a\u002a and b\u002a", "2", "1.105", "\u22120.130", "0.710", "1.135", "\u22120.196", "0.647"],
        ["<b>a\u002a alone</b>", "1", "1.091", "\u22120.084", "0.688", "<b>1.118</b>", "<b>\u22120.128</b>", "<b>0.686</b>"],
        ["<i>baseline</i>", "\u2014", "<i>1.083</i>", "<i>\u22120.082</i>", "<i>0.722</i>", "<i>1.083</i>", "<i>\u22120.082</i>", "<i>0.722</i>"],
    ], [38 * mm, 8 * mm, 20 * mm, 21 * mm, 19 * mm, 20 * mm, 21 * mm, 19 * mm]))
    story.append(Spacer(1, 3))
    story.append(para(
        "Illumination-invariant features helped materially: the erythema index lifted pipeline A from "
        "R\u00b2 \u22120.166 to \u22120.094 and F1 from 0.688 to 0.727. Both pipelines nonetheless remain at "
        "or below the baseline."
    ))

    story.append(para("6.4 Full metric comparison \u2014 models fitted on all 26 patients", "h2"))
    story.append(para(
        "Both pipelines were fitted on every patient, which is standard for a model that ships, and "
        "scored on the same patients. These figures therefore describe the <i>fit</i>; Section 6.5 gives "
        "the held-out estimate of behaviour on a new patient."
    ))
    story.append(table([
        ["Metric", "Pipeline A<br/>gray-world + erythema", "Pipeline B<br/>CIELAB + a\u002a", "Baseline"],
        ["MSE (g/dL)\u00b2", "<b>1.627</b>", "1.858", "1.877"],
        ["RMSE (g/dL)", "<b>1.275</b>", "1.363", "1.370"],
        ["MAE (g/dL)", "<b>0.944</b>", "1.038", "1.041"],
        ["R\u00b2", "<b>0.133</b>", "0.010", "0.000"],
        ["Bias (g/dL)", "0.000", "0.000", "0.000"],
        ["Pearson r", "<b>0.381</b>", "0.100", "\u2014"],
        ["Accuracy", "<b>0.654</b>", "0.615", "0.615"],
        ["Precision", "<b>0.600</b>", "0.565", "0.565"],
        ["Recall", "0.923", "<b>1.000</b>", "1.000"],
        ["Specificity", "<b>0.385</b>", "0.231", "0.231"],
        ["F1", "<b>0.727</b>", "0.722", "0.722"],
        ["Confusion", "TP 12, FP 8, FN 1, TN 5", "TP 13, FP 10, FN 0, TN 3", "\u2014"],
    ], [30 * mm, 50 * mm, 48 * mm, 40 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "<b>Pipeline A leads B on 10 of the 11 metrics.</b> B\u2019s only lead is recall, which it "
        "achieves by flagging every patient as anaemic, hence its specificity of 0.231 against "
        "A\u2019s 0.385. <b>This table should not be read as either pipeline beating the baseline.</b> "
        "In-sample it cannot be: the baseline is the full-sample mean, whose R\u00b2 is 0 by definition, "
        "and a fitted model with a free intercept essentially cannot fall below it. Measured instead "
        "against a permutation null \u2014 haemoglobin shuffled against the same features, refitted "
        "20,000 times \u2014 two free parameters on 26 patients score a mean R\u00b2 of +0.075, so "
        "A\u2019s +0.133 lies inside that null (p = 0.172) and B\u2019s +0.010 falls below the "
        "one-parameter noise mean of +0.041. A also carries one more free parameter than B here, which "
        "inflates its in-sample margin: adjusted R\u00b2 is +0.058 against \u22120.031, and matched at "
        "one feature each the pair is +0.060 against +0.010. A leads on every matched framing, but by "
        "less than the raw table suggests."
    ))
    story.append(callout(
        "<b>Why the baseline R\u00b2 is exactly 0 in this table.</b> R\u00b2 is "
        "1 &minus; &Sigma;(predicted&minus;true)&sup2; / &Sigma;(true&minus;mean)&sup2;. When the "
        "prediction <i>is</i> the full-sample mean, numerator and denominator are identical and "
        "R\u00b2 = 0 by construction \u2014 the correct comparison for a model also fitted on all "
        "patients.<br/><br/>"
        "The leave-one-out baseline in the next section is a different quantity: it must predict "
        "patient <i>i</i> from the mean of the <i>other</i> 25, which is systematically pulled away from "
        "that patient\u2019s value. Its error is inflated by exactly n/(n&minus;1), giving "
        "R\u00b2 = 1 &minus; (26/25)&sup2; = &minus;0.0816. That figure is an artefact of the protocol "
        "rather than a property of the data, which is why a fitted model and a leave-one-out baseline "
        "must never appear in the same table."))

    story.append(para("6.5 The same models on unseen patients", "h2"))
    story.append(table([
        ["Metric", "Pipeline A", "Pipeline B", "Baseline"],
        ["MAE (g/dL)", "<b>1.062</b>", "1.118", "1.083"],
        ["R\u00b2", "\u22120.094", "\u22120.128", "<b>\u22120.082</b>"],
        ["Accuracy", "<b>0.654</b>", "0.577", "0.615"],
        ["F1", "<b>0.727</b>", "0.686", "0.722"],
    ], [42 * mm, 42 * mm, 42 * mm, 42 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "<b>The gap between these two tables is the overfitting.</b> Pipeline A\u2019s R\u00b2 falls from "
        "+0.133 fitted to \u22120.094 held out: the model finds a relationship among patients it has seen "
        "that does not transfer to new ones. The fitted figures describe the model; the held-out figures "
        "describe what to expect clinically."
    ))
    story.append(para(
        "<b>Why the held-out R\u00b2 is negative while MAE beats the baseline.</b> R\u00b2 divides by the "
        "variance about the <i>full-sample</i> mean, but under leave-one-out the baseline may only use "
        "the mean of the other 25 patients. That gap is why the baseline itself scores \u22120.082 rather "
        "than 0 \u2014 the practical floor is about \u22120.08, and pipeline A sits essentially on it. On "
        "unseen patients the model is <b>statistically indistinguishable from predicting the cohort "
        "mean</b>, not catastrophically wrong."
    ))

    story.append(para("6.6 The honest number: nested cross-validation", "h2"))
    story.append(para(
        "Choosing the best representation from the table above is selection on the test set, so that "
        "figure is optimistic. A nested loop removes the bias: the inner loop picks the representation "
        "from training patients only, the outer loop scores a patient that choice never saw."
    ))
    story.append(table([
        ["", "MAE", "R\u00b2", "F1", "Representation chosen across folds"],
        ["Pipeline A", "1.117", "\u22120.210", "0.645", "erythema 14, chroma 6, means 5, a\u002a 1"],
        ["Pipeline B", "1.142", "\u22120.199", "0.647", "a\u002a 23, a\u002a/b\u002a 3"],
        ["<i>Baseline</i>", "<i>1.083</i>", "<i>\u22120.082</i>", "<i>0.722</i>", "\u2014"],
    ], [24 * mm, 20 * mm, 22 * mm, 18 * mm, 84 * mm]))
    story.append(Spacer(1, 3))
    story.append(para(
        "Under honest selection <b>neither pipeline beats the baseline</b>. Pipeline A\u2019s inner loop "
        "also disagrees with itself, flipping between four representations across 26 folds \u2014 "
        "instability that is itself evidence no representation is clearly best at this sample size."
    ))

    story.append(para("6.7 Are the two pipelines separable?", "h2"))
    story.append(para(
        "Both pipelines process the same patients, so their errors are <b>paired</b>. A paired test "
        "cancels the patient-to-patient variation they share and asks only whether one is consistently "
        "better on the same individuals \u2014 far more powerful than comparing two independent intervals."
    ))
    story.append(table([
        ["Measure", "Result"],
        ["Patients where A has the lower error", "<b>15 of 26 (58%)</b>"],
        ["Mean advantage of A", "+0.0554 g/dL"],
        ["Bootstrap 95% confidence interval", "<b>[\u22120.0665, +0.1807]</b>"],
        ["Paired permutation test", "<b>p = 0.408</b>"],
    ], [80 * mm, 88 * mm]))
    story.append(Spacer(1, 4))
    story.append(callout(
        "<b>The interval spans zero and p = 0.408: the two pipelines are not statistically separable on "
        "26 patients.</b> Choosing between them on these numbers alone would be selecting on noise. "
        "Pipeline A is nonetheless the better provisional default, for reasons independent of this test: "
        "it leads B on 10 of 11 metrics fitted, 9 of 11 held out, and on every parameter-matched "
        "framing in Section 6.4; it leads in every earlier experiment as well (cohort correlation "
        "+0.30 vs +0.02, eye-to-eye consistency 2.02 vs 2.69, lighting robustness 1.63 vs 2.74); its "
        "specificity is higher at equal recall, meaning fewer unnecessary confirmatory blood tests; and "
        "decisively, <b>B\u2019s segmentation lands on non-conjunctiva tissue in 46 of 52 real "
        "captures</b> \u2014 measured rather than eyeballed. The redness index inside A\u2019s mask "
        "has a median of +9.53 and never falls off-tissue; inside B\u2019s it is \u22123.67. B\u2019s "
        "placement is statistically indistinguishable from a brightness baseline known to select "
        "sclera, and the two masks overlap by a median of exactly zero, so they are reading different "
        "parts of the photograph, "
        "so its model is reading mostly skin. That last point is visible in the overlays and does not "
        "depend on any significance test. The comparison should be repeated on ~900 patients once the "
        "segmenter is trained; the script takes minutes to re-run."))
    story.append(Spacer(1, 4))
    story.append(para(
        "The first pipeline wins on every metric in both framings, which is the fourth independent result "
        "in that direction, though at 26 patients the margin sits inside the noise. The decisive "
        "observation is different: <b>every arm has negative R\u00b2 and an F1 at or below the trivial "
        "baseline</b>. Mean colour statistics from this cohort contain no usable haemoglobin signal."
    ))
    if diagnostics_figure and diagnostics_figure.exists():
        story.append(Spacer(1, 3))
        story.append(Image(str(diagnostics_figure), width=168 * mm, height=53 * mm))
        story.append(para(
            "Deployed model on the real cohort, leave-one-patient-out. <b>A</b>: predictions span only "
            "10.6\u201312.7 g/dL while true values span 8.1\u201314.3 \u2014 a prediction spread 33% of the "
            "actual, so the model still hedges heavily toward the cohort average (it was 23% on the "
            "earlier channel-mean model, so the band has widened but not nearly enough). <b>B</b>: "
            "Bland\u2013Altman limits of \u22122.68 to +2.77 g/dL, still far too wide to screen with. "
            "<b>C</b>: residuals rise with the prediction (slope +0.46, down from +2.02), the signature "
            "of regression toward the mean \u2014 weaker than before, not absent.", "caption"))
    story.append(Spacer(1, 3))
    story.append(callout(
        "<b>Feature richness is not the constraint \u2014 data quantity is.</b> Expanding from 3 features to "
        "24 did not help the regularised model and destroyed the unregularised one: mean absolute error "
        "11.4 g/dL and R\u00b2 \u2212118.9, a prediction wilder than the entire clinical range. With 24 "
        "parameters and 25 training points the model can nearly interpolate its training data, so it fits "
        "noise. This is the governing warning for the network stage, where roughly 8.4 million parameters "
        "will be trainable: the cohort must be pooled with the public datasets before any network result "
        "on real photographs can be believed."))

    story.append(para("6.8 The deployed model, stated explicitly", "h2"))
    story.append(para(
        "Two linear models appear in this report and it matters which is which. The <b>deployed</b> "
        "model is a ridge regression on the erythema index, fitted on all 26 patients with the quality "
        "gate applied first, and is what <font face='Courier'>runs/linear_model.json</font> contains:"
    ))
    story.append(callout(
        "Hb = 9.8308 + 15.101390 &middot; log(R/G) &minus; 12.790747 &middot; log(R/B)"
        "&nbsp;&nbsp;[g/dL]<br/><br/>"
        "Standardised: Hb = 11.2038 + 1.0490&middot;z(log(R/G)) &minus; 0.9678&middot;z(log(R/B)). "
        "The intercept 11.2038 is the cohort mean haemoglobin \u2014 with every input at its average, "
        "the model returns the average."))
    story.append(Spacer(1, 4))
    story.append(table([
        ["Evaluation", "MAE (g/dL)", "R\u00b2"],
        ["In-sample \u2014 fitted and scored on all 26", "<b>0.918</b>", "<b>+0.195</b>"],
        ["Held out \u2014 leave-one-patient-out, unseen patients", "1.033", "<b>+0.007</b>"],
        ["Baseline \u2014 leave-one-out mean", "1.083", "\u22120.082"],
    ], [96 * mm, 36 * mm, 36 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "These figures differ slightly from \u00a76.4 because fitting applies the quality gate and drops "
        "three blurred captures, which the head-to-head comparison deliberately does not \u2014 that "
        "comparison requires both pipelines to see identical patients. <b>Applying the gate before "
        "fitting is what lifted held-out R\u00b2 above zero</b>, making this the only configuration "
        "reported here that beats its baseline on patients it has not seen."
    ))
    story.append(para(
        "An <b>earlier model over channel means</b> \u2014 mean R, G and B inside the mask \u2014 is "
        "retained below because its failure produced a diagnostic still in use. It is not deployed:"
    ))
    story.append(callout(
        "Hb = 12.5311 + 0.010191 &times; (mean R) &minus; 0.058732 &times; (mean G) "
        "+ 0.035552 &times; (mean B)&nbsp;&nbsp;[g/dL]<br/><br/>"
        "Standardised: Hb = 11.2038 + 0.1373&middot;z(R) &minus; 0.9771&middot;z(G) + "
        "0.6473&middot;z(B). Worked example, patient 1: R = 156.13, G = 164.55, B = 188.01 gives "
        "11.142 g/dL against a laboratory value of 12.0 g/dL."))
    story.append(para("6.9 A physiological check on the coefficients", "h2"))
    story.append(para(
        "Haemoglobin makes tissue red, so a model reading physiology should be driven by a "
        "<b>dominant, positive red term</b>. That is a check on the fitted equation which does not "
        "depend on any accuracy metric, and it can be applied to either model above. The channel-mean "
        "model fails it:"
    ))
    story.append(table([
        ["Input", "Effect of a one-standard-deviation increase", "Share of the cohort Hb sd"],
        ["mean G", "<b>lowers</b> predicted Hb by 0.977 g/dL", "71%"],
        ["mean B", "raises predicted Hb by 0.647 g/dL", "47%"],
        ["mean R", "raises predicted Hb by 0.137 g/dL", "10%"],
    ], [26 * mm, 92 * mm, 50 * mm]))
    story.append(Spacer(1, 4))
    story.append(callout(
        "<b>If the model had found real physiology, the red channel should dominate and be positive</b> "
        "\u2014 haemoglobin makes tissue red. Instead red is the weakest of the three, and the model is "
        "driven by green negatively and blue positively, effectively a blue-minus-green contrast. That "
        "is not a haemoglobin signal; it resembles residual colour cast the white balance did not fully "
        "remove. The coefficients therefore corroborate the metrics from an independent direction: "
        "nothing physiological was learned by that representation."))
    story.append(Spacer(1, 4))
    story.append(para(
        "<b>The deployed model passes the same check.</b> Its dominant standardised coefficient is "
        "<b>+1.049 on log(R/G)</b> \u2014 red over green, positive, rising with haemoglobin, which is "
        "the sign physiology predicts; log(R/B) carries \u22120.968. The check was written down when it "
        "failed and was re-applied unchanged when the representation changed, and it reversed. A "
        "diagnostic that changes its answer when the underlying model changes is measuring something "
        "real, which is the reason for reporting both outcomes rather than only the current one. It "
        "remains the first thing to inspect after the segmenter is trained on the larger corpus \u2014 "
        "before the metrics."))

    story.append(para("6.10 Protocol: training, and why R\u00b2 is negative", "h2"))
    story.append(para(
        "Leave-one-patient-out is training: 26 models are fitted, each on 25 patients and tested on the "
        "one it never saw. Whether training succeeded is answered by comparing in-sample against "
        "out-of-sample performance."
    ))
    story.append(table([
        ["Model", "In-sample R\u00b2", "Out-of-sample R\u00b2"],
        ["Channel means, no quality gate", "+0.103", "\u22120.166"],
        ["Erythema index + quality gate (deployed)", "<b>+0.195</b>", "<b>+0.007</b>"],
    ], [96 * mm, 36 * mm, 36 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "In-sample R\u00b2 is positive in both rows, so the models do fit a relationship; <b>the gap "
        "between the two columns is the overfitting.</b> R\u00b2 is bounded between 0 and 1 only when a "
        "model is scored on the data it was fitted to; on held-out data it may go below zero, and a "
        "negative value simply means the model performs worse than ignoring the image and predicting the "
        "cohort average. The finding is not an artefact of the protocol: for the channel-mean model, "
        "5-fold cross-validation gives \u22120.175 and 13-fold gives \u22120.168."
    ))
    story.append(para(
        "<b>One caution on reading any in-sample figure in this report.</b> Comparing a model fitted on "
        "all 26 patients against a predict-the-mean baseline on those same patients is a comparison the "
        "model cannot lose: the baseline\u2019s R\u00b2 is exactly 0 by definition, and a fitted model "
        "with a free intercept essentially cannot fall below it. The reference that carries information "
        "is what <i>noise</i> scores. Shuffling haemoglobin against the same features and refitting, "
        "20,000 times, two free parameters on 26 patients reach a mean R\u00b2 of <b>+0.075</b>. "
        "Measured against that null, the deployed model\u2019s +0.195 gives <b>p = 0.074</b> in-sample "
        "and <b>p = 0.067</b> held out \u2014 the strongest result obtained so far, and still short of "
        "significance at this sample size. The head-to-head figures in \u00a76.4 sit further inside the "
        "null (p = 0.172 for pipeline A). Adjusted for the number of free parameters, pipeline A\u2019s "
        "in-sample R\u00b2 is +0.058 against pipeline B\u2019s \u22120.031."
    ))
    story.append(para(
        "There is deliberately <b>no separate validation set</b>. A 60/20/20 split of 26 patients leaves "
        "roughly 15 training, 5 validation and 6 test. Repeating that split under 200 random seeds on "
        "identical data and an identical model gives a test R\u00b2 ranging from <b>\u22122.43 to +0.78</b> "
        "and a test MAE from 0.32 to 2.26 g/dL \u2014 a single split at this sample size reports which "
        "patients landed where, not model quality. Accordingly no hyperparameter is tuned: the ridge "
        "penalty is a fixed default rather than a fitted value, because choosing it on the evaluation "
        "set would leak. A three-way split becomes both possible and correct once the corpus reaches "
        "~900 images, which is also when leave-one-out stops being affordable for a network."
    ))

    story.append(para("6.11 Uncertainty on every reported number", "h2"))
    story.append(para(
        "A point estimate from 26 patients is not precise, and quoting it alone overstates certainty. "
        "Bootstrap resampling over patients, 2000 draws:"
    ))
    story.append(table([
        ["Metric", "Point estimate", "95% confidence interval"],
        ["MAE (g/dL)", "1.073", "<b>[0.708, 1.490]</b>"],
        ["R\u00b2", "\u22120.166", "<b>[\u22120.512, +0.021]</b>"],
        ["F1", "0.688", "<b>[0.462, 0.850]</b>"],
    ], [56 * mm, 56 * mm, 56 * mm]))
    story.append(Spacer(1, 4))
    story.append(para(
        "The R\u00b2 interval reaches just above zero, so the defensible statement is <b>\u201cno better "
        "than predicting the mean\u201d</b> rather than a precise \u22120.166. Every figure in this report "
        "should be read with intervals of this width in mind."
    ))

    story.append(para("6.12 Strengths and limitations of this evaluation", "h2"))
    story.append(table([
        ["Strength", "Limitation"],
        ["Every metric is paired with an explicit baseline, so a model that has learned nothing cannot "
         "appear competitive", "26 patients; confidence intervals comfortably exceed the differences "
         "between arms"],
        ["Splits are grouped by patient, so two captures of one subject cannot straddle train and test",
         "Leave-one-out gives a nearly unbiased but high-variance error estimate"],
        ["Leave-one-out uses every patient as a test case, maximising scarce data",
         "The regularisation strength is fixed rather than tuned, because tuning on the evaluation set "
         "would leak"],
        ["Regression and screening are scored separately, so a good g/dL error cannot hide a poor "
         "clinical decision", "Features are channel means; a network sees spatial structure that these "
         "summaries discard"],
        ["The same code path scores synthetic and real data, so the comparison is like for like",
         "One cohort, one device, one operator"],
    ], [84 * mm, 84 * mm]))

    # ------------------------------------------------------------- status
    story.append(para("7. Verification status", "h1"))
    story.append(para("7.1 Verified", "h2"))
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
        "A complete metric suite is implemented and reproducible \u2014 MSE, RMSE, MAE, median absolute "
        "error, bias, R\u00b2, Bland\u2013Altman limits, accuracy, precision, recall, specificity, NPV, F1 and "
        "balanced accuracy \u2014 each reported beside its baseline.",
        "Colour correction helps consistently: four independent measurements (cohort correlation, "
        "eye-to-eye consistency, lighting stress, representation ablation) all favour gray-world.",
    ]))
    story.append(para("7.2 Measured and found wanting", "h2"))
    story.extend(bullets([
        "The plain colour threshold is unreliable on real tissue, bleeding onto eyelashes and lid "
        "skin. A refined seeded-grabCut extractor built against the same captures now produces clean "
        "masks on visual review across all 52 (Section 4.3); a measured Dice still awaits ground-truth "
        "masks.",
        "The blur threshold looks slightly too aggressive for real captures: two of three rejections "
        "were marginal on images that appear usable.",
        "Quality control does not yet test illuminant neutrality, so a strongly tinted capture passes "
        "and yields a confident but meaningless number.",
        "No linear model over colour statistics beats predict-the-mean on the 26-patient cohort "
        "(Section 6.3); the linear floor is unmet.",
    ]))
    story.append(para("7.3 Not yet established", "h2"))
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
    story.append(para("8. Data", "h1"))
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
    story.append(para("9. Risks", "h1"))
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
    story.append(para("10. Next steps", "h1"))
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
        ["5", "Add an illuminant-neutrality check to quality control, and recalibrate the blur threshold",
         "Rejects tinted and marginal captures instead of scoring them"],
        ["6", "Pool the local cohort with the public datasets once segmentation is reliable",
         "26 balanced, well-labelled patients added to training"],
        ["7", "Re-run Sections 6.2\u20136.3 on the trained network against the same baselines",
         "First defensible accuracy claim on real photographs"],
        ["8", "Build the application around the existing API", "Working demonstrator"],
    ], [8 * mm, 78 * mm, 82 * mm]))

    story.append(Spacer(1, 8))
    story.append(callout(
        "<b>Method record.</b> Every approach tried, including those rejected, is recorded with its "
        "reason in <font face='Courier'>RESEARCH_LOG.md</font> \u2014 six method attempts that failed, ten "
        "defects found and fixed, four null results, and seven approaches deliberately not attempted. "
        "<br/><br/><b>Assessment.</b> The pipeline is in a state where real data can be introduced and produce "
        "trustworthy numbers, has been exercised on real photographs rather than synthetic ones, and now "
        "reports a full metric suite against explicit baselines. "
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
    parser.add_argument("--diagnostics-figure", type=Path, default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.out, args.figure, args.real_figure, args.diagnostics_figure)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
