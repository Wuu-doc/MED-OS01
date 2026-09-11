#!/usr/bin/env python3
"""Generate manuscript Figures 2 and 1 with one shared visual language."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# -----------------------------------------------------------------------------
# Shared journal-style design system (used without alteration by both figures)
# -----------------------------------------------------------------------------
FONT_FAMILY = "DejaVu Sans"
BOX_EDGE = "#3F464D"
BOX_FACE = "#F7F8F8"
HEADER_FACE = "#DDE4E8"
ACCENT_FACE = "#E8EEF1"
SECONDARY_FACE = "#F0F2F3"
TEXT_COLOR = "#202428"
MUTED_TEXT = "#4C545B"
ARROW_COLOR = "#4A5157"
LINE_WIDTH = 1.15
ARROW_WIDTH = 1.15
TITLE_FONTSIZE = 10.2
BODY_FONTSIZE = 8.5
SMALL_FONTSIZE = 7.4
BOX_ROUNDING = 0.009
BOX_PAD = 0.003

OUTPUT_DIR = Path(__file__).resolve().parent


def new_canvas(figsize: tuple[float, float]):
    """Return the common white, normalized-coordinate drawing canvas."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def draw_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    text: str | None = None,
    facecolor: str = BOX_FACE,
    edgecolor: str = BOX_EDGE,
    linewidth: float = LINE_WIDTH,
    fontsize: float = BODY_FONTSIZE,
    fontweight: str = "normal",
    text_color: str = TEXT_COLOR,
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.28,
    zorder: int = 2,
):
    """Draw a rounded box using normalized lower-left coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size={BOX_ROUNDING}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        mutation_aspect=1,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        tx = x + width / 2 if ha == "center" else x + 0.012
        ax.text(
            tx,
            y + height / 2,
            text,
            ha=ha,
            va=va,
            color=text_color,
            fontsize=fontsize,
            fontweight=fontweight,
            family=FONT_FAMILY,
            linespacing=linespacing,
            zorder=zorder + 1,
        )
    return patch


def draw_header(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = HEADER_FACE,
    fontsize: float = TITLE_FONTSIZE,
    linespacing: float = 1.28,
):
    """Draw the shared emphasized header box."""
    return draw_box(
        ax,
        x,
        y,
        width,
        height,
        text=text,
        facecolor=facecolor,
        fontsize=fontsize,
        fontweight="bold",
        linespacing=linespacing,
    )


def draw_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = BODY_FONTSIZE,
    fontweight: str = "normal",
    color: str = TEXT_COLOR,
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.25,
    zorder: int = 5,
):
    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        family=FONT_FAMILY,
        linespacing=linespacing,
        zorder=zorder,
    )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3,rad=0",
    arrowstyle: str = "-|>",
    linewidth: float = ARROW_WIDTH,
    color: str = ARROW_COLOR,
    mutation_scale: float = 12,
    zorder: int = 1,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        connectionstyle=connectionstyle,
        linewidth=linewidth,
        color=color,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_split_arrow(
    ax,
    source: tuple[float, float],
    left_target: tuple[float, float],
    right_target: tuple[float, float],
    *,
    branch_y: float,
):
    """Draw a clean orthogonal one-to-two split without crossing text."""
    sx, sy = source
    lx, ly = left_target
    rx, ry = right_target
    draw_arrow(ax, (sx, sy), (sx, branch_y), arrowstyle="-", zorder=1)
    draw_arrow(ax, (lx, branch_y), (rx, branch_y), arrowstyle="-", zorder=1)
    draw_arrow(ax, (lx, branch_y), (lx, ly), zorder=1)
    draw_arrow(ax, (rx, branch_y), (rx, ry), zorder=1)


def draw_merge_arrow(
    ax,
    left_source: tuple[float, float],
    right_source: tuple[float, float],
    target: tuple[float, float],
    *,
    merge_y: float,
):
    """Draw a clean orthogonal two-to-one merge."""
    lx, ly = left_source
    rx, ry = right_source
    tx, ty = target
    draw_arrow(ax, (lx, ly), (lx, merge_y), arrowstyle="-", zorder=1)
    draw_arrow(ax, (rx, ry), (rx, merge_y), arrowstyle="-", zorder=1)
    draw_arrow(ax, (lx, merge_y), (rx, merge_y), arrowstyle="-", zorder=1)
    draw_arrow(ax, (tx, merge_y), (tx, ty), zorder=1)


def draw_vertical_sequence(ax, center_x: float, boxes: list[tuple[float, float]]):
    """Connect (bottom_y, height) boxes from top to bottom."""
    for (upper_y, _), (lower_y, lower_h) in zip(boxes, boxes[1:]):
        draw_arrow(ax, (center_x, upper_y), (center_x, lower_y + lower_h))


def save_figure(fig, stem: str) -> tuple[Path, Path]:
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


# -----------------------------------------------------------------------------
# Figure 2: micro-level leakage-free evaluation workflow (generated first)
# -----------------------------------------------------------------------------
def generate_figure2() -> tuple[Path, Path]:
    fig, ax = new_canvas((14, 8))

    # Top-level inputs and validation design.
    draw_header(ax, 0.415, 0.925, 0.170, 0.050, "Dataset")
    draw_arrow(ax, (0.500, 0.925), (0.500, 0.886))
    draw_box(
        ax,
        0.290,
        0.810,
        0.420,
        0.076,
        text=(
            "Repeated Stratified 10-Fold Cross-Validation × 5 Repetitions\n"
            "(50 held-out evaluations per model–sampling configuration)"
        ),
        facecolor=ACCENT_FACE,
        fontsize=8.9,
        fontweight="bold",
    )

    # Fold split.
    left_x, right_x, branch_w = 0.105, 0.595, 0.300
    left_c, right_c = left_x + branch_w / 2, right_x + branch_w / 2
    draw_split_arrow(
        ax,
        (0.500, 0.810),
        (left_c, 0.755),
        (right_c, 0.755),
        branch_y=0.775,
    )
    draw_header(ax, left_x, 0.710, branch_w, 0.045, "Training fold")
    draw_header(ax, right_x, 0.710, branch_w, 0.045, "Held-out fold")

    left_boxes = [
        (0.610, 0.067, "Missing-value imputation", BOX_FACE),
        (0.510, 0.067, "Standardization", BOX_FACE),
        (
            0.410,
            0.067,
            "Class-imbalance handling within training fold only\nBaseline / SMOTE / ADASYN",
            SECONDARY_FACE,
        ),
        (
            0.310,
            0.067,
            "Train classifier\nLR / RF / MLP / XGBoost / LightGBM",
            ACCENT_FACE,
        ),
    ]
    right_boxes = [
        (
            0.610,
            0.067,
            "Apply imputation parameters\nlearned from training fold",
            BOX_FACE,
        ),
        (
            0.510,
            0.067,
            "Apply standardization parameters\nlearned from training fold",
            BOX_FACE,
        ),
        (0.410, 0.067, "No oversampling", SECONDARY_FACE),
        (0.310, 0.067, "Generate predicted probabilities", ACCENT_FACE),
    ]
    for y, h, text, face in left_boxes:
        draw_box(ax, left_x, y, branch_w, h, text=text, facecolor=face)
    for y, h, text, face in right_boxes:
        draw_box(ax, right_x, y, branch_w, h, text=text, facecolor=face)
    draw_vertical_sequence(ax, left_c, [(0.710, 0.045)] + [(y, h) for y, h, _, _ in left_boxes])
    draw_vertical_sequence(ax, right_c, [(0.710, 0.045)] + [(y, h) for y, h, _, _ in right_boxes])

    # The trained model enters the held-out prediction operation horizontally.
    draw_arrow(ax, (left_x + branch_w, 0.3435), (right_x, 0.3435))
    draw_text(
        ax,
        0.500,
        0.357,
        "trained model",
        fontsize=SMALL_FONTSIZE,
        color=MUTED_TEXT,
        va="bottom",
    )

    # Predicted probabilities feed two scientifically distinct evaluations.
    # Only the left branch applies decision thresholds; ranking/calibration
    # metrics on the right connect directly to the predicted probabilities.
    left_eval_x, left_eval_w = 0.055, 0.445
    right_eval_x, right_eval_w = 0.545, 0.400
    left_eval_c = left_eval_x + left_eval_w / 2
    right_eval_c = right_eval_x + right_eval_w / 2
    draw_split_arrow(
        ax,
        (right_c, 0.310),
        (0.440, 0.267),
        (0.860, 0.267),
        branch_y=0.285,
    )
    draw_box(
        ax,
        left_eval_x,
        0.190,
        left_eval_w,
        0.077,
        text=(
            "Apply predefined decision thresholds to the same predicted probabilities\n"
            "0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70"
        ),
        facecolor=ACCENT_FACE,
        fontsize=7.6,
        fontweight="bold",
    )
    draw_box(
        ax,
        right_eval_x,
        0.202,
        right_eval_w,
        0.065,
        text="Threshold-independent and calibration evaluation",
        facecolor=ACCENT_FACE,
        fontsize=8.0,
        fontweight="bold",
    )
    draw_arrow(ax, (left_eval_c, 0.190), (left_eval_c, 0.169))
    draw_arrow(ax, (right_eval_c, 0.202), (right_eval_c, 0.169))
    draw_box(
        ax,
        0.100,
        0.088,
        0.355,
        0.081,
        text=(
            "Threshold-dependent metrics\n"
            "F1-score · Precision\n"
            "Sensitivity · Specificity · Balanced Accuracy"
        ),
        fontsize=7.7,
    )
    draw_box(
        ax,
        0.590,
        0.088,
        0.310,
        0.081,
        text="ROC-AUC\nAverage Precision\nBrier score",
        fontsize=8.0,
    )
    draw_merge_arrow(
        ax,
        (left_eval_c, 0.088),
        (right_eval_c, 0.088),
        (0.500, 0.065),
        merge_y=0.074,
    )
    draw_box(
        ax,
        0.350,
        0.010,
        0.300,
        0.055,
        text="Aggregate results across\n50 held-out evaluations",
        facecolor=HEADER_FACE,
        fontsize=8.7,
        fontweight="bold",
    )

    return save_figure(fig, "Figure2_leakage_free_workflow")


# -----------------------------------------------------------------------------
# Figure 1: macro-level overall research framework (same design system)
# -----------------------------------------------------------------------------
def draw_major_block(ax, x: float, width: float, title: str):
    """Draw a Figure 1 major block and its header using shared primitives."""
    draw_box(ax, x, 0.085, width, 0.830, facecolor="white")
    title_linespacing = 0.92 if title == "2. Stage 1\nPrimary\nBenchmark" else 1.28
    draw_header(
        ax,
        x + 0.006,
        0.838,
        width - 0.012,
        0.069,
        title,
        fontsize=8.8,
        linespacing=title_linespacing,
    )


def generate_figure1() -> tuple[Path, Path]:
    fig, ax = new_canvas((15, 7.5))

    gap = 0.012
    widths = [0.140, 0.110, 0.300, 0.195, 0.195]
    xs = [0.006]
    for width in widths[:-1]:
        xs.append(xs[-1] + width + gap)

    titles = [
        "1. Research\nObjective",
        "2. Stage 1\nPrimary\nBenchmark",
        "3. Unified Experimental\nFramework",
        "4. Stage 2\nCross-Dataset Validation",
        "5. Cross-Dataset\nAnalysis",
    ]
    for x, width, title in zip(xs, widths, titles):
        draw_major_block(ax, x, width, title)

    # Major left-to-right flow arrows remain entirely within the gaps.
    for x, width, next_x in zip(xs, widths, xs[1:]):
        draw_arrow(
            ax,
            (x + width + 0.001, 0.500),
            (next_x - 0.001, 0.500),
            mutation_scale=10,
            linewidth=1.0,
            zorder=4,
        )

    # Block 1: objective.
    draw_text(
        ax,
        xs[0] + widths[0] / 2,
        0.760,
        "Systematically examine\nthe effects of:",
        fontsize=8.2,
        fontweight="bold",
    )
    draw_text(
        ax,
        xs[0] + 0.016,
        0.560,
        "• Oversampling method\n"
        "• Oversampling intensity\n"
        "• Classification model\n"
        "• Decision threshold",
        fontsize=7.9,
        ha="left",
        linespacing=1.55,
    )
    draw_arrow(
        ax,
        (xs[0] + widths[0] / 2, 0.492),
        (xs[0] + widths[0] / 2, 0.448),
        mutation_scale=10,
        linewidth=1.0,
        zorder=4,
    )
    draw_box(
        ax,
        xs[0] + 0.012,
        0.310,
        widths[0] - 0.024,
        0.130,
        facecolor=ACCENT_FACE,
    )
    draw_text(
        ax,
        xs[0] + widths[0] / 2,
        0.407,
        "Outcome",
        fontsize=7.2,
        fontweight="bold",
    )
    draw_text(
        ax,
        xs[0] + widths[0] / 2,
        0.354,
        "Imbalanced medical\nclassification performance",
        fontsize=7.2,
        fontweight="bold",
    )

    # Block 2: primary benchmark.
    draw_box(
        ax,
        xs[1] + 0.012,
        0.285,
        widths[1] - 0.024,
        0.355,
        facecolor=ACCENT_FACE,
    )
    draw_text(
        ax,
        xs[1] + widths[1] / 2,
        0.545,
        "PIMA",
        fontsize=11.0,
        fontweight="bold",
    )
    draw_text(
        ax,
        xs[1] + widths[1] / 2,
        0.460,
        "Primary\nbenchmark\ndataset",
        fontsize=8.2,
        fontweight="bold",
    )
    draw_text(
        ax,
        xs[1] + widths[1] / 2,
        0.365,
        "Establish\nthe unified\nexperimental\nframework",
        fontsize=SMALL_FONTSIZE,
    )

    # Block 3: the visually largest, grouped unified framework.
    x3, w3 = xs[2], widths[2]
    inner_x, inner_w = x3 + 0.012, w3 - 0.024
    draw_box(
        ax,
        inner_x,
        0.736,
        inner_w,
        0.080,
        text=(
            "Validation\nRepeated Stratified 10-Fold CV × 5 Repetitions"
        ),
        facecolor=ACCENT_FACE,
        fontsize=7.7,
        fontweight="bold",
    )
    draw_box(
        ax,
        inner_x,
        0.634,
        inner_w,
        0.080,
        text=(
            "Leakage control\nPreprocessing and oversampling\nwithin training folds only"
        ),
        facecolor=SECONDARY_FACE,
        fontsize=7.1,
        fontweight="bold",
    )
    col_gap = 0.010
    col_w = (inner_w - col_gap) / 2
    draw_box(
        ax,
        inner_x,
        0.432,
        col_w,
        0.180,
        text=(
            "Sampling conditions\nBaseline / SMOTE / ADASYN\n\n"
            "Oversampling intensity\nCommon comparison:\n0.50 / 0.75 / 1.00\n"
            "Low-intensity sensitivity:\nSMOTE = 0.25"
        ),
        fontsize=7.0,
    )
    draw_box(
        ax,
        inner_x + col_w + col_gap,
        0.432,
        col_w,
        0.180,
        text=(
            "Classification models\nLR / RF / MLP\nXGBoost / LightGBM\n\n"
            "Decision thresholds\nPredefined: 0.30–0.70\nstep = 0.05"
        ),
        fontsize=7.1,
    )
    draw_box(
        ax,
        inner_x,
        0.114,
        inner_w,
        0.294,
        text=(
            "Evaluation\n\nPrimary: F1-score\n\nSupporting:\n"
            "Sensitivity · Specificity · Precision\nBalanced Accuracy · ROC-AUC\n"
            "AP · Brier score"
        ),
        facecolor=BOX_FACE,
        fontsize=7.5,
    )

    # Block 4: validation datasets are clearly parallel, not transferred models.
    x4, w4 = xs[3], widths[3]
    sub_gap = 0.010
    sub_w = (w4 - 0.036 - sub_gap) / 2
    draw_text(
        ax,
        x4 + w4 / 2,
        0.772,
        "Additional validation datasets",
        fontsize=8.0,
        fontweight="bold",
    )
    draw_box(
        ax,
        x4 + 0.012,
        0.438,
        sub_w,
        0.265,
        text=(
            "Heart Failure\n\n299 samples\nPositive class\n= 32.1%\n\nModerate\nimbalance"
        ),
        facecolor=ACCENT_FACE,
        fontsize=7.3,
        fontweight="bold",
    )
    draw_box(
        ax,
        x4 + 0.012 + sub_w + sub_gap,
        0.438,
        sub_w,
        0.265,
        text=(
            "Thoracic Surgery\n\n470 samples\nPositive class\n= 14.9%\n\nSevere\nimbalance"
        ),
        facecolor=ACCENT_FACE,
        fontsize=7.3,
        fontweight="bold",
    )
    draw_box(
        ax,
        x4 + 0.020,
        0.190,
        w4 - 0.040,
        0.155,
        text=(
            "Apply the same core\nexperimental protocol\nindependently"
        ),
        facecolor=SECONDARY_FACE,
        fontsize=8.0,
        fontweight="bold",
    )

    # Block 5: cross-dataset analyses and appropriately qualified interpretation.
    x5, w5 = xs[4], widths[4]
    draw_text(
        ax,
        x5 + 0.014,
        0.620,
        "• Oversampling-intensity and\n  decision-threshold effects\n"
        "• Intensity × threshold interaction\n"
        "• Matched SMOTE–ADASYN\n  comparison\n"
        "• Configuration-level consistency\n  and factor-wise transferability\n"
        "• Calibration assessment",
        fontsize=6.85,
        ha="left",
        linespacing=1.38,
    )
    draw_box(
        ax,
        x5 + 0.018,
        0.320,
        w5 - 0.036,
        0.085,
        text="Robustness\nSMOTENC\nHyperparameter tuning",
        facecolor=SECONDARY_FACE,
        fontsize=6.9,
        fontweight="bold",
    )
    draw_box(
        ax,
        x5 + 0.012,
        0.150,
        w5 - 0.024,
        0.135,
        text=(
            "Final interpretation\n\nDataset-specific configuration\n"
            "rather than a universally\noptimal setting"
        ),
        facecolor=HEADER_FACE,
        fontsize=7.5,
        fontweight="bold",
    )

    return save_figure(fig, "Figure1_overall_research_framework")


def run_quality_control(outputs: list[Path]) -> None:
    required_figure2_text = (
        "Repeated Stratified 10-Fold Cross-Validation × 5 Repetitions",
        "50 held-out evaluations per model–sampling configuration",
        "Class-imbalance handling within training fold only",
        "No oversampling",
        "Threshold-dependent metrics",
        "Threshold-independent and calibration evaluation",
        "ROC-AUC",
        "Average Precision",
        "Brier score",
        "0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70",
    )
    required_figure1_text = (
        "PIMA",
        "Heart Failure",
        "Thoracic Surgery",
        "Predefined: 0.30–0.70",
        "SMOTE = 0.25",
        "independently",
    )
    # These literals make accidental content loss fail loudly during maintenance.
    source = Path(__file__).read_text(encoding="utf-8")
    for phrase in required_figure2_text + required_figure1_text:
        if phrase not in source:
            raise RuntimeError(f"Required scientific content is missing: {phrase}")
    prohibited = ("threshold " + "optimization", "optimized " + "threshold")
    if any(phrase in source.lower() for phrase in prohibited):
        raise RuntimeError("Prohibited threshold-optimization wording detected")
    for output in outputs:
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--figure2-only",
        action="store_true",
        help="Regenerate only Figure 2, leaving existing Figure 1 files untouched.",
    )
    selection.add_argument(
        "--figure1-only",
        action="store_true",
        help="Regenerate only Figure 1, leaving existing Figure 2 files untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.figure1_only:
        figure1_png, figure1_pdf = generate_figure1()
        run_quality_control([figure1_png, figure1_pdf])
        print("QA PASS")
        print(f"Figure 1 PNG: {figure1_png} ({png_dimensions(figure1_png)[0]} × {png_dimensions(figure1_png)[1]} px)")
        print(f"Figure 1 PDF: {figure1_pdf}")
        return
    # Required execution order: establish Figure 2 first, then reuse its style.
    figure2_png, figure2_pdf = generate_figure2()
    if args.figure2_only:
        run_quality_control([figure2_png, figure2_pdf])
        print("QA PASS")
        print(f"Figure 2 PNG: {figure2_png} ({png_dimensions(figure2_png)[0]} × {png_dimensions(figure2_png)[1]} px)")
        print(f"Figure 2 PDF: {figure2_pdf}")
        return
    figure1_png, figure1_pdf = generate_figure1()
    outputs = [figure1_png, figure1_pdf, figure2_png, figure2_pdf]
    run_quality_control(outputs)

    print("QA PASS")
    print(f"Figure 1 PNG: {figure1_png} ({png_dimensions(figure1_png)[0]} × {png_dimensions(figure1_png)[1]} px)")
    print(f"Figure 1 PDF: {figure1_pdf}")
    print(f"Figure 2 PNG: {figure2_png} ({png_dimensions(figure2_png)[0]} × {png_dimensions(figure2_png)[1]} px)")
    print(f"Figure 2 PDF: {figure2_pdf}")
    print("Figure 2 generated first; both figures use the shared style and drawing helpers.")


if __name__ == "__main__":
    main()
