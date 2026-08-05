from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "ctrg"
FIGURES = ROOT / "elsarticle" / "figures"
TABLES = ROOT / "elsarticle" / "tables"
YEARS = (2023, 2024, 2025)

METHOD_LABELS = {
    "structured_multitask_relational_model": "KCCRES relation model",
    "observed_path_risk": "Observed-path risk",
    "kccres_max_gap_certificate": "Maximum recovery contrast",
    "kccres_dual_channel_certificate": "Joint risk--contrast",
    "risk_gap_support_certificate": "Risk--contrast--support",
}

METHOD_STYLES = {
    "structured_multitask_relational_model": ("#245f8d", "-", "o"),
    "observed_path_risk": ("#6f6f6f", (0, (2, 1)), "s"),
    "kccres_max_gap_certificate": ("#c17a16", (0, (5, 2)), "^"),
    "kccres_dual_channel_certificate": ("#6e4b8b", (0, (4, 1, 1, 1)), "D"),
    "risk_gap_support_certificate": ("#278678", (0, (1, 1)), "P"),
}

METHOD_TOP10_KEYS = {
    "structured_multitask_relational_model": "Structured multitask relational model",
    "observed_path_risk": "Observed-path risk",
    "kccres_max_gap_certificate": "KCCRES max-gap certificate",
    "kccres_dual_channel_certificate": "KCCRES dual-channel certificate",
    "risk_gap_support_certificate": "Risk-gap-support certificate",
}

TOP10_LABELS = {
    "Structured multitask relational model": "KCCRES relation model",
    "Observed-path risk": "Observed-path risk",
    "KCCRES max-gap certificate": "Maximum recovery contrast",
    "KCCRES dual-channel certificate": "Joint risk--contrast",
    "Risk-gap-support certificate": "Risk--contrast--support",
}

ANNUAL_SCALE = {
    2023: {"turnarounds": 5_336_525, "supported": 428_008, "support": 0.8831},
    2024: {"turnarounds": 5_432_367, "supported": 431_399, "support": 0.8864},
    2025: {"turnarounds": 5_303_729, "supported": 479_211, "support": 0.8873},
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_inputs() -> dict:
    data: dict = {"annual": {}, "prior": {}, "calibration": {}}
    for year in YEARS:
        transport_dir = RESULTS / f"final_transport_{year}_full"
        model_dir = RESULTS / f"final_multitask_{year}_full"
        calibration_dir = RESULTS / f"calibration_{year}"
        data["annual"][year] = {
            "report": json.loads((transport_dir / "report.json").read_text(encoding="utf-8"))["experiments"][0],
            "frontier": pd.read_csv(transport_dir / "capacity_frontier.csv"),
            "top10": pd.read_csv(model_dir / "validation_top10.csv"),
        }
        data["calibration"][year] = json.loads(
            (calibration_dir / "report.json").read_text(encoding="utf-8")
        )
    for year, score_dir in (
        (2023, "prior_record_2023_multitask_full"),
        (2025, "final_prior_record_multitask_2025_full"),
    ):
        certificate_dir = RESULTS / f"final_nested_drift_prior_record_{year}_full"
        data["prior"][year] = {
            "report": json.loads((certificate_dir / "report.json").read_text(encoding="utf-8")),
            "frontier": pd.read_csv(certificate_dir / "nested_capacity_frontier.csv"),
            "top10": pd.read_csv(RESULTS / score_dir / "validation_top10.csv"),
        }
    return data


def validate_inputs(data: dict) -> None:
    required_frontier = {"capacity", "method", "transported_risk_upper_bound"}
    required_top10 = {"method", "opportunity_precision", "opportunity_capture"}
    for year in YEARS:
        annual = data["annual"][year]
        if not required_frontier.issubset(annual["frontier"].columns):
            raise ValueError(f"Missing annual frontier columns for {year}")
        if not required_top10.issubset(annual["top10"].columns):
            raise ValueError(f"Missing annual top-10 columns for {year}")
        report = annual["report"]
        if not report["gate_passed"] or not report["support_passed"]:
            raise ValueError(f"Final annual gate is not passed for {year}")
    for year in (2023, 2025):
        prior = data["prior"][year]
        required = {"capacity", "method", "drift_radius", "fixed_sequence_certified"}
        if not required.issubset(prior["frontier"].columns):
            raise ValueError(f"Missing prior-record frontier columns for {year}")
        if not prior["report"]["proposed_certificate_passed"]:
            raise ValueError(f"Final prior-record certificate is not passed for {year}")


def write_table(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text.strip() + "\n", encoding="utf-8")


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def tex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def table_system_interface() -> None:
    write_table(
        "tab_system_interface.tex",
        r"""
\begin{table}[!t]
\centering
\caption{KCCRES evidence, learning, certification, and action interface.}
\label{tab:system-interface}
\begin{tabularx}{\linewidth}{@{}L{0.20\linewidth}L{0.28\linewidth}Y L{0.18\linewidth}@{}}
\toprule
Layer & Object & Decision meaning & Favorable condition \\
\midrule
CTRG evidence & Compatible donors and Wilson recovery lower bound & Establishes whether an observed recovery alternative has adequate empirical support & More supported focal episodes and a higher lower bound \\
Relational learning & Focal failure, feasible-alternative support, and joint recovery opportunity & Concentrates cases whose observed path is brittle and whose compatible histories recover & Higher opportunity precision and capture \\
Completed-record certificate & Score-conditional Clopper--Pearson mixture bound $U_{\mathrm{tr}}(c)$ & Selects a review fraction after simultaneous finite-sample risk control & Smaller certified $c$ with $U_{\mathrm{tr}}(c)\le\rho$ \\
Prior-record certificate & Fixed-sequence bound $U_{\mathrm{CP}}(c)+\varepsilon+\eta$ & Authorizes clearance under a declared aggregate shift budget & Bound below $\rho$ with adequate residual support \\
Action layer & Review, automatic clearance, or abstention & Routes cases and exposes the certificate and donor audit trail & Residual risk below target; abstention outside support \\
\bottomrule
\end{tabularx}
\tablenote{CTRG denotes the Compatible Tail-Recovery Graph. KCCRES denotes the Knowledge- and Capacity-Certified Recoverability Expert System.}
\end{table}
""",
    )


def table_annual_data(data: dict) -> None:
    support_rows = []
    calibration_rows = []
    for year in YEARS:
        metrics = {m["period"]: m for m in data["calibration"][year]["metrics"]}
        validation = metrics["validation_calibrated"]
        scale = ANNUAL_SCALE[year]
        support_rows.append(
            f"{year} & {tex_int(scale['turnarounds'])} & {tex_int(scale['supported'])} & "
            f"{pct(scale['support'])} \\\\"
        )
        calibration_rows.append(
            f"{year} & {validation['auc']:.3f} & {validation['average_precision']:.3f} & "
            f"{validation['brier_score']:.4f} & {pct(validation['expected_calibration_error'])} \\\\"
        )
    write_table(
        "tab_annual_data_support_calibration.tex",
        r"""
\begin{table}[!t]
\centering
\setlength{\tabcolsep}{5pt}
\caption{Annual operational scale and evidence support.}
\label{tab:annual-data-support-calibration}
\begin{tabular}{@{}lrrr@{}}
\toprule
Year & Reconstructed turnarounds & Supported stressed & Support \\
\midrule
""" + "\n".join(support_rows) + r"""
\bottomrule
\end{tabular}
\tablenote{Higher support is favorable because it expands the population for which compatible-continuation inference is available.}
\end{table}
""",
    )
    write_table(
        "tab_annual_discrimination_calibration.tex",
        r"""
\begin{table}[!t]
\centering
\setlength{\tabcolsep}{6pt}
\caption{Annual observed-path discrimination and calibration.}
\label{tab:annual-discrimination-calibration}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Year & AUC & AP & Brier score & ECE \\
\midrule
""" + "\n".join(calibration_rows) + r"""
\bottomrule
\end{tabular}
\tablenote{Higher area under the receiver operating characteristic curve (AUC) and average precision (AP) are favorable. Lower Brier score and expected calibration error (ECE) indicate more reliable probabilities. Metrics use chronological validation segments.}
\end{table}
""",
    )


def table_method_comparison(data: dict) -> None:
    order = list(TOP10_LABELS)
    rows = []
    values = {}
    for source_method in order:
        values[source_method] = []
        captures = []
        for year in YEARS:
            row = data["annual"][year]["top10"].query("method == @source_method").iloc[0]
            values[source_method].append(float(row.opportunity_precision))
            captures.append(float(row.opportunity_capture))
        values[source_method].append(captures[-1])
    maxima = [max(values[m][j] for m in order) for j in range(4)]
    for method in order:
        cells = []
        for j, value in enumerate(values[method]):
            text = pct(value)
            if abs(value - maxima[j]) < 1e-12:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        rows.append(f"{TOP10_LABELS[method]} & " + " & ".join(cells) + r" \\")
    write_table(
        "tab_annual_method_comparison.tex",
        r"""
\begin{table}[!t]
\centering
\caption{Recovery-opportunity ranking comparison across recent operating years.}
\label{tab:annual-method-comparison}
\begin{tabularx}{\linewidth}{@{}Y C C C C@{}}
\toprule
Method & 2023 precision & 2024 precision & 2025 precision & 2025 capture \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\tablenote{Higher top-10\% recovery-opportunity precision and capture are favorable. Bold values mark the largest value in each column. All methods use the same focal population, donor graph, temporal split, and target.}
\end{table}
""",
    )


def load_ablation() -> list[tuple[str, float]]:
    specifications = [
        ("Complete relation model", "parallel_validation_20k"),
        ("Temporal relations removed", "ablation_20k_no_relations"),
        ("Logical and factorization penalties removed", "ablation_20k_no_logic"),
        ("Differentiable capacity term removed", "ablation_20k_no_capacity_loss"),
        ("Auxiliary tasks removed", "ablation_20k_no_multitask"),
    ]
    rows = []
    for label, directory in specifications:
        frame = pd.read_csv(RESULTS / directory / "validation_top10.csv")
        value = float(frame.query("method == 'Structured multitask relational model'").iloc[0].opportunity_precision)
        rows.append((label, value))
    return rows


def table_ablation() -> None:
    ablation = load_ablation()
    best = max(v for _, v in ablation)
    rows = []
    for label, value in ablation:
        cell = pct(value, 1)
        delta = 100.0 * (value - best)
        if value == best:
            cell = rf"\textbf{{{cell}}}"
        rows.append(f"{label} & {cell} & {delta:.1f} pp \\\\ ")
    write_table(
        "tab_component_ablation.tex",
        r"""
\begin{table}[!t]
\centering
\caption{Contribution of the relational and capacity-aware model components.}
\label{tab:component-ablation}
\begin{tabularx}{\linewidth}{@{}Y C C@{}}
\toprule
Model specification & Top-10\% opportunity precision & Change from complete \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\tablenote{Higher precision is favorable. The component study uses one fixed 20{,}000-episode frame and a held-out 5{,}000-episode validation segment. Bold marks the complete specification.}
\end{table}
""",
    )


def table_prior_methods(data: dict) -> None:
    order = list(TOP10_LABELS)
    report_keys = {
        "Structured multitask relational model": "structured_multitask_relational_model",
        "Observed-path risk": "observed_path_risk",
        "KCCRES max-gap certificate": "kccres_max_gap_certificate",
        "KCCRES dual-channel certificate": "kccres_dual_channel_certificate",
        "Risk-gap-support certificate": "risk_gap_support_certificate",
    }
    rows = []
    for source_method in order:
        cells = []
        for year in (2023, 2025):
            top = data["prior"][year]["top10"].query("method == @source_method").iloc[0]
            result = data["prior"][year]["report"]["methods"][report_keys[source_method]]
            precision = pct(float(top.opportunity_precision))
            if result["risk_control_feasible"] and result["validation_risk_passed"]:
                status = f"Pass / {pct(result['validation_residual_rate'])}"
            elif result["risk_control_feasible"]:
                status = f"Validation fail / {pct(result['validation_residual_rate'])}"
            else:
                status = "No certified capacity"
            if source_method == "Structured multitask relational model":
                precision = rf"\textbf{{{precision}}}"
                status = rf"\textbf{{{status}}}"
            cells.extend([precision, status])
        rows.append(f"{TOP10_LABELS[source_method]} & " + " & ".join(cells) + r" \\")
    write_table(
        "tab_prior_record_method_comparison.tex",
        r"""
\begin{table}[!t]
\centering
\caption{Prior-record ranking and bounded-shift certification by method.}
\label{tab:prior-record-method-comparison}
\begin{tabularx}{\linewidth}{@{}Y C L{0.20\linewidth} C L{0.20\linewidth}@{}}
\toprule
Method & 2023 precision & 2023 certificate / residual & 2025 precision & 2025 certificate / residual \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\tablenote{Higher top-10\% precision is favorable. A joint pass requires a feasible fixed-sequence certificate and held-out residual risk below target. Bold marks the only method with a joint pass in both annual evaluations.}
\end{table}
""",
    )


def figure_annual_summary(data: dict) -> None:
    x = np.arange(len(YEARS))
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.1))
    axes = axes.ravel()
    for method, label in METHOD_LABELS.items():
        color, linestyle, marker = METHOD_STYLES[method]
        precision, capture, capacity = [], [], []
        for year in YEARS:
            source_label = METHOD_TOP10_KEYS[method]
            top = data["annual"][year]["top10"].query("method == @source_label").iloc[0]
            precision.append(100 * float(top.opportunity_precision))
            capture.append(100 * float(top.opportunity_capture))
            record = next(m for m in data["annual"][year]["report"]["methods"] if m["method"] == method)
            capacity.append(100 * float(record["capacity"]) if record["risk_control_feasible"] else np.nan)
        axes[0].plot(x, precision, color=color, linestyle=linestyle, marker=marker,
                     linewidth=2.2 if method.startswith("structured") else 1.25,
                     markersize=4.7, label=label)
        axes[1].plot(x, capture, color=color, linestyle=linestyle, marker=marker,
                     linewidth=2.2 if method.startswith("structured") else 1.25, markersize=4.7)
        axes[2].plot(x, capacity, color=color, linestyle=linestyle, marker=marker,
                     linewidth=2.2 if method.startswith("structured") else 1.25, markersize=4.7)
    residual = [100 * data["annual"][y]["report"]["proposed_validation_residual_rate"] for y in YEARS]
    target = [100 * data["annual"][y]["report"]["risk_target"] for y in YEARS]
    axes[3].fill_between(x, 0, target, color="#e7f2ee", alpha=0.9, label="Admissible region")
    axes[3].plot(x, target, color="#b9770e", marker="s", linewidth=1.7, label="Risk target")
    axes[3].plot(x, residual, color="#278678", marker="o", linewidth=2.1, label="Held-out residual")
    for i, (observed, limit) in enumerate(zip(residual, target)):
        alignment = "left" if i == 0 else ("right" if i == len(YEARS) - 1 else "center")
        axes[3].annotate(f"{limit-observed:.2f} pp headroom", (i, observed), xytext=(0, -15),
                         textcoords="offset points", ha=alignment, fontsize=6.8)
    titles = ("(a) Top-10% opportunity precision", "(b) Top-10% opportunity capture",
              "(c) Risk-certified review capacity", "(d) Held-out risk control")
    ylabels = ("Precision (%)", "Capture (%)", "Review capacity (%)", "Opportunity risk (%)")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xticks(x, YEARS)
        ax.grid(axis="y", color="#dedede", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].text(1.5, 72, "Observed-path risk: infeasible in 2024/2025",
                 ha="center", fontsize=6.8, color="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    extra_h, extra_l = axes[3].get_legend_handles_labels()
    fig.legend(handles + extra_h[1:], labels + extra_l[1:], loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.005), columnspacing=1.2, handlelength=2.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.18, hspace=0.42, wspace=0.27)
    fig.savefig(FIGURES / "fig_annual_increment_capacity_risk.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_completed_frontiers(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.15), sharey=False)
    for ax, year in zip(axes, YEARS):
        annual = data["annual"][year]
        report = annual["report"]
        method_rows = {m["method"]: m for m in report["methods"]}
        ax.axhspan(0, report["risk_target"] * 100, color="#e7f2ee", alpha=0.9)
        for method, label in METHOD_LABELS.items():
            color, linestyle, marker = METHOD_STYLES[method]
            frame = annual["frontier"].query("method == @method")
            ax.plot(frame.capacity * 100, frame.transported_risk_upper_bound * 100, label=label,
                    color=color, linestyle=linestyle,
                    linewidth=2.25 if method.startswith("structured") else 1.15,
                    alpha=1.0 if method.startswith("structured") else 0.85)
            selected = method_rows[method]
            if selected["risk_control_feasible"]:
                ax.scatter(selected["capacity"] * 100, selected["transported_risk_upper_bound"] * 100,
                           color=color, marker=marker, s=28, zorder=4)
            else:
                boundary = frame.iloc[-1]
                ax.scatter(boundary.capacity * 100, boundary.transported_risk_upper_bound * 100,
                           facecolors="none", edgecolors=color, marker="X", s=32, zorder=4)
        ax.axhline(report["risk_target"] * 100, color="#b9770e", linestyle="--", linewidth=1.2, label="Risk target")
        proposed_capacity = 100 * report["proposed_capacity"]
        comparison_capacity = 100 * report["best_feasible_baseline_capacity"]
        ax.annotate(f"{proposed_capacity:.0f}% vs {comparison_capacity:.0f}%",
                    xy=(proposed_capacity, report["risk_target"] * 100), xytext=(4, 11),
                    textcoords="offset points", fontsize=6.8, color="#245f8d")
        ax.set_title(f"({chr(97 + list(YEARS).index(year))}) {year}")
        ax.set_xlabel("Review capacity (%)")
        ax.set_ylabel("Transported upper bound (%)")
        ax.set_xlim(0, 80)
        ax.grid(color="#e2e2e2", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handlelength=2.8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.28, wspace=0.34)
    fig.savefig(FIGURES / "fig_completed_record_capacity_frontiers.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_prior_frontiers(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.0), sharex="col")
    for column, year in enumerate((2023, 2025)):
        prior = data["prior"][year]
        drift_ax, risk_ax = axes[0, column], axes[1, column]
        required = prior["report"]["required_drift_radius"] * 100
        ceiling = 100 * (prior["report"]["risk_target"] - prior["report"]["required_drift_radius"]
                         - prior["report"]["safety_margin"])
        drift_ax.axhspan(required, 2.2, color="#e7f2ee", alpha=0.9)
        risk_ax.axhspan(0, ceiling, color="#e7f2ee", alpha=0.9)
        for method, label in METHOD_LABELS.items():
            color, linestyle, marker = METHOD_STYLES[method]
            frame = prior["frontier"].query("method == @method").sort_values("capacity")
            width = 2.2 if method.startswith("structured") else 1.1
            drift_ax.plot(frame.capacity * 100, frame.drift_radius * 100, color=color,
                          linestyle=linestyle, linewidth=width, label=label)
            risk_ax.plot(frame.capacity * 100, frame.risk_upper_bound * 100, color=color,
                         linestyle=linestyle, linewidth=width)
            certified = frame[frame.fixed_sequence_certified]
            if not certified.empty:
                drift_ax.scatter(certified.capacity * 100, certified.drift_radius * 100,
                                 color=color, marker=marker, s=32, zorder=4)
                risk_ax.scatter(certified.capacity * 100, certified.risk_upper_bound * 100,
                                color=color, marker=marker, s=32, zorder=4)
        drift_ax.axhline(required, color="#b9770e", linestyle="--", linewidth=1.25,
                         label="Declared drift budget")
        risk_ax.axhline(ceiling, color="#b9770e", linestyle="--", linewidth=1.25,
                        label="Admissible finite-sample bound")
        validation = 100 * prior["report"]["methods"]["structured_multitask_relational_model"]["validation_residual_rate"]
        risk_ax.text(0.97, 0.10, f"KCCRES: 2% clear; held-out risk {validation:.2f}%",
                     transform=risk_ax.transAxes, ha="right", va="bottom",
                     fontsize=6.8, color="#245f8d")
        drift_ax.set_title(f"({chr(97 + column)}) {year}: admissible shift")
        risk_ax.set_title(f"({chr(99 + column)}) {year}: finite-sample risk bound")
        drift_ax.set_ylabel("Admissible drift (pp)")
        risk_ax.set_ylabel("Risk upper bound (%)"); risk_ax.set_xlabel("Review capacity (%)")
        drift_ax.set_xlim(38, 100); drift_ax.set_ylim(top=max(1.85, drift_ax.get_ylim()[1]))
        for ax in (drift_ax, risk_ax):
            ax.grid(color="#e2e2e2", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[1, 0].lines[-1]); labels.append("Admissible finite-sample bound")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), columnspacing=1.15, handlelength=2.7)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.19, hspace=0.38, wspace=0.25)
    fig.savefig(FIGURES / "fig_prior_record_drift_frontiers.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_ablation_calibration(data: dict) -> None:
    ablation = load_ablation()
    labels = [x[0] for x in ablation]
    values = [100 * x[1] for x in ablation]
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.2))
    axes = axes.ravel()
    y = np.arange(len(labels))
    axes[0].hlines(y, 20, values, color=["#245f8d"] + ["#a7b6c2"] * (len(values) - 1), linewidth=2.0)
    axes[0].scatter(values, y, color=["#245f8d"] + ["#8799a6"] * (len(values) - 1), s=34, zorder=3)
    axes[0].set_yticks(y, ["Complete", "No relations", "No structure", "No capacity term", "No auxiliary tasks"])
    axes[0].invert_yaxis(); axes[0].set_xlim(20, 24)
    axes[0].set_xlabel("Top-10% precision (%)"); axes[0].set_title("(a) Component contribution")
    for idx, value in enumerate(values):
        suffix = "complete" if idx == 0 else f"{value-values[0]:.1f} pp"
        axes[0].text(value + 0.06, idx, suffix, va="center", fontsize=6.8)
    raw_brier, cal_brier, raw_ece, cal_ece, raw_bias, cal_bias, sample_sizes = [], [], [], [], [], [], []
    for year in YEARS:
        metrics = {m["period"]: m for m in data["calibration"][year]["metrics"]}
        raw = metrics["validation_raw"]; calibrated = metrics["validation_calibrated"]
        raw_brier.append(raw["brier_score"]); cal_brier.append(calibrated["brier_score"])
        raw_ece.append(raw["expected_calibration_error"] * 100); cal_ece.append(calibrated["expected_calibration_error"] * 100)
        raw_bias.append(abs(raw["mean_probability"] - raw["recovery_prevalence"]) * 100)
        cal_bias.append(abs(calibrated["mean_probability"] - calibrated["recovery_prevalence"]) * 100)
        sample_sizes.append(int(calibrated["episodes"]))
    metrics_to_plot = (
        (raw_brier, cal_brier, "(b) Brier score", "Brier score"),
        (raw_ece, cal_ece, "(c) Expected calibration error", "Error (%)"),
        (raw_bias, cal_bias, "(d) Mean-probability bias", "Absolute bias (pp)"),
    )
    row_y = np.arange(3)
    for ax, (raw_values, calibrated_values, title, xlabel) in zip(axes[1:], metrics_to_plot):
        for yi, raw_value, calibrated_value in zip(row_y, raw_values, calibrated_values):
            ax.plot([calibrated_value, raw_value], [yi, yi], color="#a7b6c2", linewidth=2.0, zorder=1)
        ax.scatter(raw_values, row_y, color="#8799a6", marker="s", s=32, label="Raw", zorder=3)
        ax.scatter(calibrated_values, row_y, color="#278678", marker="o", s=34, label="Calibrated", zorder=3)
        ax.set_yticks(row_y, [f"{year}  n={sample_sizes[i]:,}" for i, year in enumerate(YEARS)])
        ax.set_ylim(2.55, -0.55); ax.set_title(title); ax.set_xlabel(xlabel)
        for yi, raw_value, calibrated_value in zip(row_y, raw_values, calibrated_values):
            reduction = 100 * (raw_value - calibrated_value) / raw_value
            vertical_offset = 11 if yi == row_y[-1] else -11
            ax.annotate(f"-{reduction:.0f}%", (calibrated_value, yi), xytext=(0, vertical_offset),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=6.6, color="#278678", clip_on=True)
    for ax in axes:
        ax.grid(axis="x", color="#e2e2e2", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axes[1], axes[3]):
        ax.set_yticks(row_y, YEARS)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", labelleft=False, labelright=True, pad=3)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.16, right=0.94, top=0.93, bottom=0.15, hspace=0.45, wspace=0.44)
    fig.savefig(FIGURES / "fig_ablation_and_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def build(data: dict) -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    table_system_interface()
    table_annual_data(data)
    table_method_comparison(data)
    table_ablation()
    table_prior_methods(data)
    figure_annual_summary(data)
    figure_completed_frontiers(data)
    figure_prior_frontiers(data)
    figure_ablation_calibration(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data = load_inputs()
    validate_inputs(data)
    print("Final display-input evaluation check passed for 2023--2025 and prior-record 2023/2025.")
    if not args.check_only:
        build(data)
        print("Generated four data figures and five main-text tables from final result files.")


if __name__ == "__main__":
    main()
