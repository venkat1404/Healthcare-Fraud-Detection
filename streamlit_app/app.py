from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_processing import DEMO_DATA_DIR, PROCESSED_DATA_DIR
from src.decision_engine import DEFAULT_THRESHOLDS, apply_decision_engine, summarize_decision_actions, thresholds_are_valid
from src.generate_demo_data import save_demo_claims
from src.predict import get_missing_synthetic_scoring_columns, score_synthetic_claims_with_context

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


st.set_page_config(page_title="AI Healthcare Fraud Detection System", layout="wide")

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
DEMO_FILE = DEMO_DATA_DIR / "demo_scored_claims.csv"

PALETTE = {
    "background": "#0f0f13",
    "navy": "#e2e8f0",
    "blue": "#6366f1",
    "green": "#22c55e",
    "amber": "#f59e0b",
    "orange": "#f97316",
    "red": "#f43f5e",
    "critical": "#e11d48",
    "card": "#1a1a24",
    "muted": "#94a3b8",
    "border": "#2e2e3e",
}

RISK_COLORS = {
    "Low Risk": PALETTE["green"],
    "Medium Risk": PALETTE["amber"],
    "High Risk": PALETTE["red"],
    "Borderline Risk": PALETTE["amber"],
    "Elevated Risk": PALETTE["amber"],
    "Critical Risk": PALETTE["red"],
}

DECISION_COLORS = {
    "Auto Approve": PALETTE["green"],
    "Auto Approve After Secondary Check": PALETTE["green"],
    "Secondary Rules Check": PALETTE["amber"],
    "Human Review Queue": PALETTE["amber"],
    "Priority Investigation": PALETTE["orange"] if "orange" in PALETTE else "#EA580C",
    "Payment Hold + Mandatory Investigator Review": PALETTE["critical"] if "critical" in PALETTE else "#991B1B",
}


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {PALETTE["background"]};
            color: {PALETTE["navy"]};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(15,23,42,0.04));
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
        }}
        .hero-title {{
            color: {PALETTE["navy"]};
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .hero-subtitle {{
            color: {PALETTE["muted"]};
            font-size: 1rem;
            line-height: 1.5;
        }}
        .section-title {{
            color: {PALETTE["navy"]};
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0.2rem 0 0.8rem 0;
        }}
        .subsection-title {{
            color: {PALETTE["navy"]};
            font-size: 1rem;
            font-weight: 700;
            margin: 0.2rem 0 0.6rem 0;
        }}
        .insight-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 16px;
            padding: 1rem 1rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            height: 100%;
        }}
        .section-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
        }}
        .kpi-card {{
            background: {PALETTE["card"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .kpi-label {{
            color: {PALETTE["muted"]};
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .kpi-value {{
            color: {PALETTE["navy"]};
            font-size: 1.45rem;
            font-weight: 700;
            white-space: normal;
            overflow-wrap: break-word;
            line-height: 1.3;
        }}
        .kpi-sub {{
            color: {PALETTE["muted"]};
            font-size: 0.75rem;
            margin-top: 0.2rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .pill {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.3rem;
        }}
        .note-box {{
            background: {PALETTE["card"]};
            border-left: 4px solid {PALETTE["blue"]};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            color: {PALETTE["navy"]};
            margin: 0.6rem 0 1rem 0;
        }}
        .small-text {{
            color: {PALETTE["muted"]};
            font-size: 0.84rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin: 0.15rem 0.25rem 0.15rem 0;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1a1a24 0%, #0f0f13 100%);
            border-right: 1px solid {PALETTE["border"]};
        }}
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCheckbox label {{
            color: {PALETTE["navy"]};
        }}
        [data-testid="stExpander"] {{
            border: 1px solid {PALETTE["border"]};
            border-radius: 14px;
            background: #1a1a24;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def _render_subsection_title(title: str) -> None:
    st.markdown(f'<div class="subsection-title">{title}</div>', unsafe_allow_html=True)


def _render_kpi(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_currency(value: float, compact: bool = False) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    if compact:
        abs_value = abs(value)
        if abs_value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        if abs_value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        if abs_value >= 1_000:
            return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"


def _format_probability(value: float) -> str:
    if pd.isna(value):
        return "0.0%"
    return f"{float(value):.1%}"


def _format_yes_no(value) -> str:
    return "Yes" if bool(value) else "No"


def _badge_html(label: str, kind: str = "risk") -> str:
    color_map = RISK_COLORS if kind == "risk" else DECISION_COLORS
    bg = color_map.get(label, PALETTE["blue"])
    return f'<span class="badge" style="background:{bg}1A;color:{bg};border:1px solid {bg}40;">{label}</span>'


def _render_badge_row(items: List[Tuple[str, str]]) -> None:
    html = "".join(_badge_html(label, kind) for label, kind in items)
    st.markdown(html, unsafe_allow_html=True)


def _display_table(
    df: pd.DataFrame,
    columns_map: Dict[str, str],
    money_columns: Iterable[str] | None = None,
    probability_columns: Iterable[str] | None = None,
    bool_columns: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    money_columns = set(money_columns or [])
    probability_columns = set(probability_columns or [])
    bool_columns = set(bool_columns or [])

    available = [column for column in columns_map if column in df.columns]
    table = df[available].copy()
    if max_rows is not None:
        table = table.head(max_rows)

    for column in available:
        if column in money_columns:
            table[column] = table[column].apply(_format_currency)
        elif column in probability_columns:
            table[column] = table[column].apply(_format_probability)
        elif column in bool_columns:
            table[column] = table[column].apply(_format_yes_no)

    table = table.rename(columns=columns_map)
    st.dataframe(table, use_container_width=True, hide_index=True)
    return table


def _apply_plotly_layout(fig, title: str | None = None, height: int = 360):
    layout_kwargs = dict(
        height=height,
        plot_bgcolor="#1a1a24",
        paper_bgcolor="#1a1a24",
        margin=dict(l=20, r=20, t=50 if title else 10, b=20),
        legend_title_text="",
        font=dict(color=PALETTE["navy"]),
        hoverlabel=dict(bgcolor="#1a1a24", font_color="#e2e8f0"),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=PALETTE["navy"]))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
    fig.update_yaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
    return fig


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> Dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_demo_data() -> pd.DataFrame:
    if not DEMO_FILE.exists():
        save_demo_claims(output_path=DEMO_FILE, num_records=180, seed=42)
    return pd.read_csv(DEMO_FILE)


def _safe_metric(metrics: Dict, key: str, percentage: bool = False) -> str:
    value = metrics.get(key)
    if value is None:
        return "N/A"
    if percentage:
        return f"{value:.1%}"
    return f"{value:.3f}"


def _normalize_risk_label(value: str) -> str:
    text = str(value).strip()
    mapping = {
        "Low Risk / Auto Approve": "Low Risk",
        "Medium Risk / Human Review": "Medium Risk",
        "High Risk / Investigate Before Payment": "High Risk",
        "Low Risk": "Low Risk",
        "Medium Risk": "Medium Risk",
        "High Risk": "High Risk",
    }
    return mapping.get(text, text)


def _normalize_action_label(value: str, probability: float | None = None) -> str:
    text = str(value).strip()
    mapping = {
        "Auto approve with routine monitoring.": "Auto Approve",
        "Send to human reviewer before payment.": "Human Review",
        "Investigate before payment and require human review before denial.": "Investigate Before Payment",
        "Human verification required because model confidence is below 80%.": "Human Review",
        "Low Risk / Auto Approve": "Auto Approve",
        "Medium Risk / Human Review": "Human Review",
        "High Risk / Investigate Before Payment": "Investigate Before Payment",
    }
    if text in mapping:
        return mapping[text]
    if text in {"Auto Approve", "Human Review", "Investigate Before Payment"}:
        return text
    if probability is None:
        return text
    if probability < 0.30:
        return "Auto Approve"
    if probability < 0.70:
        return "Human Review"
    return "Investigate Before Payment"


def _human_review_required(probability: float, claim_amount: float) -> bool:
    return probability >= 0.30 or claim_amount > 10000


def _derive_top_risk_driver(row: pd.Series) -> str:
    driver_scores = {
        "High reimbursement amount": (
            row.get("Reimbursed_Amount", 0) / max(row.get("Claim_Amount", 1), 1)
        ),
        "Unusually high claim frequency": row.get("Claim_Count_Last_12_Months", 0) / 150,
        "Prior provider fraud history": 1.25 if bool(row.get("Prior_Fraud_Flag", False)) else 0.0,
        "Many unique diagnosis codes": row.get("Unique_Diagnosis_Codes", 0) / 12,
        "Many unique procedure codes": row.get("Unique_Procedure_Codes", 0) / 8,
        "High deductible amount": (
            row.get("Deductible_Amount", 0) / max(row.get("Claim_Amount", 1), 1)
        ) * 4,
        "Complex chronic condition profile": row.get("Chronic_Condition_Count", 0) / 8,
    }
    best_driver = max(driver_scores, key=driver_scores.get)
    if driver_scores[best_driver] < 0.2:
        return "Routine claim pattern"
    return best_driver


def _column_or_default(df: pd.DataFrame, column: str, default_value):
    if column in df.columns:
        return df[column]
    return pd.Series([default_value] * len(df), index=df.index)


def _standardize_dashboard_frame(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()

    if "fraud_probability" in standardized.columns and "Model_Fraud_Probability" not in standardized.columns:
        standardized["Model_Fraud_Probability"] = standardized["fraud_probability"]
    if "risk_category" in standardized.columns and "Risk_Category" not in standardized.columns:
        standardized["Risk_Category"] = standardized["risk_category"]
    if "recommended_action" in standardized.columns and "Recommended_Action" not in standardized.columns:
        standardized["Recommended_Action"] = standardized["recommended_action"]
    if "model_confidence" in standardized.columns and "Model_Confidence" not in standardized.columns:
        standardized["Model_Confidence"] = standardized["model_confidence"]

    if "Claim_ID" not in standardized.columns:
        standardized["Claim_ID"] = [f"UPLOAD_{index + 1:04d}" for index in range(len(standardized))]
    if "Provider_ID" not in standardized.columns:
        standardized["Provider_ID"] = standardized.get("Provider", "Unknown Provider")
    if "Patient_ID" not in standardized.columns:
        standardized["Patient_ID"] = [f"PATGEN_{index + 1:05d}" for index in range(len(standardized))]
    if "Claim_Type" not in standardized.columns:
        standardized["Claim_Type"] = standardized.get("Service_Type", "Unknown")
    if "Region" not in standardized.columns:
        standardized["Region"] = standardized.get("Provider_State", "Unknown")
    if "Provider_Type" not in standardized.columns:
        standardized["Provider_Type"] = "Unknown"
    if "Claim_Amount" not in standardized.columns:
        standardized["Claim_Amount"] = standardized.get("Reimbursed_Amount", 0)
    if "Reimbursed_Amount" not in standardized.columns:
        standardized["Reimbursed_Amount"] = standardized.get("Claim_Amount", 0)
    if "Deductible_Amount" not in standardized.columns:
        standardized["Deductible_Amount"] = standardized.get("DeductibleAmtPaid", 0)
    if "Claim_Count_Last_12_Months" not in standardized.columns:
        standardized["Claim_Count_Last_12_Months"] = standardized.get("Number_of_Previous_Claims_Provider", 0)
    if "Unique_Patients_Last_12_Months" not in standardized.columns:
        standardized["Unique_Patients_Last_12_Months"] = standardized.get("Number_of_Previous_Claims_Patient", 0)
    if "Unique_Diagnosis_Codes" not in standardized.columns:
        standardized["Unique_Diagnosis_Codes"] = 0
    if "Unique_Procedure_Codes" not in standardized.columns:
        standardized["Unique_Procedure_Codes"] = 0
    if "Chronic_Condition_Count" not in standardized.columns:
        standardized["Chronic_Condition_Count"] = 0
    if "Prior_Fraud_Flag" not in standardized.columns:
        standardized["Prior_Fraud_Flag"] = False

    standardized["Model_Fraud_Probability"] = pd.to_numeric(
        _column_or_default(standardized, "Model_Fraud_Probability", 0), errors="coerce"
    ).fillna(0)
    standardized["Risk_Category"] = _column_or_default(standardized, "Risk_Category", "Low Risk").apply(_normalize_risk_label)
    standardized["Recommended_Action"] = [
        _normalize_action_label(action, prob)
        for action, prob in zip(
            _column_or_default(standardized, "Recommended_Action", ""),
            standardized["Model_Fraud_Probability"],
        )
    ]

    if "Top_Risk_Driver" not in standardized.columns:
        standardized["Top_Risk_Driver"] = standardized.apply(_derive_top_risk_driver, axis=1)
    standardized["Human_Review_Required"] = [
        bool(value) if pd.notna(value) else _human_review_required(prob, amount)
        for value, prob, amount in zip(
            _column_or_default(standardized, "Human_Review_Required", None),
            standardized["Model_Fraud_Probability"],
            pd.to_numeric(standardized["Claim_Amount"], errors="coerce").fillna(0),
        )
    ]

    return standardized


def _current_thresholds() -> Dict[str, float]:
    for key, value in DEFAULT_THRESHOLDS.items():
        st.session_state.setdefault(key, value)
    return {
        "low_upper": float(st.session_state["low_upper"]),
        "borderline_upper": float(st.session_state["borderline_upper"]),
        "elevated_upper": float(st.session_state["elevated_upper"]),
        "high_upper": float(st.session_state["high_upper"]),
    }


def _reset_thresholds() -> None:
    for key, value in DEFAULT_THRESHOLDS.items():
        st.session_state[key] = value


def _apply_engine_safe(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    standardized = _standardize_dashboard_frame(df)
    if "Model_Fraud_Probability" in standardized.columns and thresholds_are_valid(thresholds):
        return apply_decision_engine(standardized, thresholds=thresholds)
    return standardized


def _empty_dashboard_message(page_name: str) -> None:
    st.markdown(
        f"""
        <div class="note-box">
            No dataset is currently loaded for <strong>{page_name}</strong>.<br><br>
            Go to <strong>Batch Scoring</strong> and upload a CSV to populate the dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _placeholder_model_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "logistic_regression", "accuracy": 0.88, "precision": 0.44, "recall": 0.87, "f1": 0.59, "roc_auc": 0.95},
            {"model": "random_forest", "accuracy": 0.94, "precision": 0.75, "recall": 0.52, "f1": 0.61, "roc_auc": 0.95},
        ]
    )


def _filtered_batch_frame(
    df: pd.DataFrame,
    selected_risks: Iterable[str],
    threshold: float,
    top_n: int,
    selected_regions: Iterable[str],
    selected_claim_types: Iterable[str],
) -> pd.DataFrame:
    filtered = df.copy()
    filtered = filtered[filtered["Risk_Category"].isin(list(selected_risks))]
    filtered = filtered[filtered["Model_Fraud_Probability"] >= threshold]
    if selected_regions and "Region" in filtered.columns:
        filtered = filtered[filtered["Region"].isin(list(selected_regions))]
    if selected_claim_types and "Claim_Type" in filtered.columns:
        filtered = filtered[filtered["Claim_Type"].isin(list(selected_claim_types))]
    return filtered.sort_values("Model_Fraud_Probability", ascending=False).head(top_n)


def _render_plotly_or_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> None:
    if df.empty:
        st.info("No data available for this chart yet.")
        return
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            color_discrete_map=RISK_COLORS if color else None,
            title=title,
        )
        bar_layout = dict(
            plot_bgcolor="#1a1a24",
            paper_bgcolor="#1a1a24",
            margin=dict(l=20, r=20, t=50 if title else 10, b=20),
            legend_title_text="",
            font=dict(color=PALETTE["navy"]),
        )
        if title:
            bar_layout["title"] = dict(text=title, font=dict(color=PALETTE["navy"]))
        fig.update_layout(**bar_layout)
        fig.update_xaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
        fig.update_yaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        plot_frame = df.set_index(x)[y]
        st.bar_chart(plot_frame)


def _render_case_feature_chart(case_row: pd.Series) -> None:
    feature_frame = pd.DataFrame(
        [
            {"feature": "Claim Amount", "score": min(case_row["Claim_Amount"] / 15000, 1.0)},
            {"feature": "Claim Frequency", "score": min(case_row["Claim_Count_Last_12_Months"] / 160, 1.0)},
            {"feature": "Diagnosis Variety", "score": min(case_row["Unique_Diagnosis_Codes"] / 12, 1.0)},
            {"feature": "Procedure Variety", "score": min(case_row["Unique_Procedure_Codes"] / 8, 1.0)},
            {"feature": "Chronic Complexity", "score": min(case_row["Chronic_Condition_Count"] / 8, 1.0)},
            {"feature": "Prior Fraud Flag", "score": 1.0 if bool(case_row["Prior_Fraud_Flag"]) else 0.0},
        ]
    ).sort_values("score", ascending=True)

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            feature_frame,
            x="score",
            y="feature",
            orientation="h",
            color="score",
            color_continuous_scale=["#312e81", "#6366f1", "#f43f5e"],
        )
        fig.update_layout(
            title="Potential Contributing Factors",
            plot_bgcolor="#1a1a24",
            paper_bgcolor="#1a1a24",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(color=PALETTE["navy"]),
        )
        fig.update_xaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
        fig.update_yaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(feature_frame.set_index("feature")["score"])


_inject_styles()

provider_metrics = _load_json(str(METRICS_DIR / "provider_best_model_metrics.json"))
synthetic_metrics = _load_json(str(METRICS_DIR / "synthetic_best_model_metrics.json"))
provider_leaderboard = _load_csv(str(METRICS_DIR / "provider_model_leaderboard.csv"))
synthetic_leaderboard = _load_csv(str(METRICS_DIR / "synthetic_model_leaderboard.csv"))
provider_thresholds = _load_csv(str(METRICS_DIR / "provider_threshold_metrics.csv"))
synthetic_thresholds = _load_csv(str(METRICS_DIR / "synthetic_threshold_metrics.csv"))
provider_importance = _load_csv(str(METRICS_DIR / "provider_feature_importance.csv"))
thresholds = _current_thresholds()
st.session_state.setdefault("loaded_dataset", pd.DataFrame())
st.session_state.setdefault("current_data_source", "No Data Loaded")
st.session_state.setdefault("current_record_count", 0)
st.session_state.setdefault("current_human_reviews", 0)
st.session_state.setdefault("current_payment_holds", 0)

loaded_dataset = st.session_state["loaded_dataset"]
demo_df = _apply_engine_safe(loaded_dataset, thresholds) if not loaded_dataset.empty else pd.DataFrame()

st.sidebar.title("Command Center")
presentation_mode = st.sidebar.checkbox("Presentation Mode", value=False)
st.sidebar.markdown("### Status")
st.sidebar.markdown(
    f"""
    <div class="insight-card" style="padding:0.85rem;">
        <div class="small-text"><strong>Source</strong></div>
        <div style="font-weight:700;color:{PALETTE["navy"]};margin-bottom:0.45rem;">{st.session_state["current_data_source"]}</div>
        <div class="small-text">Total records: <strong>{st.session_state["current_record_count"]:,}</strong></div>
        <div class="small-text">Human reviews required: <strong>{st.session_state["current_human_reviews"]:,}</strong></div>
        <div class="small-text">Payment holds: <strong>{st.session_state["current_payment_holds"]:,}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Pages",
    [
        "Overview",
        "Batch Scoring",
        "Decision Engine",
        "Case Review",
        "Business Impact",
        "Responsible AI",
    ],
    label_visibility="collapsed",
)

with st.sidebar.expander("📊 Technical Reference"):
    secondary_page = st.radio(
        "Secondary",
        ["Unselect", "Model Insights", "About"],
        label_visibility="collapsed",
    )
    if secondary_page != "Unselect":
        page = secondary_page

if page == "Overview":
    _render_header(
        "AI-Powered Healthcare Insurance Fraud Detection System",
        "Executive command center for healthcare fraud triage. Upload a dataset in Batch Scoring to populate this dashboard for walkthroughs and team review.",
    )
    if demo_df.empty:
        _empty_dashboard_message("Overview")
        st.stop()

    total_cases = len(demo_df)
    high_risk_cases = int((demo_df["Risk_Category"] == "High Risk").sum())
    medium_risk_cases = int((demo_df["Risk_Category"] == "Medium Risk").sum())
    low_risk_cases = int((demo_df["Risk_Category"] == "Low Risk").sum())
    avg_probability = demo_df["Model_Fraud_Probability"].mean()
    estimated_exposure = demo_df.loc[
        demo_df["Risk_Category"] == "High Risk", "Reimbursed_Amount"
    ].sum()
    decision_summary = summarize_decision_actions(demo_df)

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.9rem;margin-bottom:0.9rem;">
            <div class="kpi-card">
                <div class="kpi-label">Total Claims Scored</div>
                <div class="kpi-value">{total_cases:,}</div>
                <div class="kpi-sub">Preloaded demo command set</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">High-Risk Cases</div>
                <div class="kpi-value">{high_risk_cases:,}</div>
                <div class="kpi-sub">Immediate investigator focus</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Human Reviews Required</div>
                <div class="kpi-value">{decision_summary["Total Human Reviews Required"]:,}</div>
                <div class="kpi-sub">Investigator workload</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Estimated Fraud Exposure</div>
                <div class="kpi-value">{_format_currency(estimated_exposure, compact=True)}</div>
                <div class="kpi-sub">High-risk reimbursed amount</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.9rem;margin-bottom:1.2rem;">
            <div class="kpi-card">
                <div class="kpi-label">Auto Approved</div>
                <div class="kpi-value">{decision_summary["Auto Approve"]:,}</div>
                <div class="kpi-sub">Routed by automation layer</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Payment Holds</div>
                <div class="kpi-value">{decision_summary["Payment Holds"]:,}</div>
                <div class="kpi-sub">Critical-risk routing</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Priority Investigations</div>
                <div class="kpi-value">{decision_summary["Priority Investigation"]:,}</div>
                <div class="kpi-sub">High-risk queue</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Fraud Probability</div>
                <div class="kpi-value">{avg_probability:.1%}</div>
                <div class="kpi-sub">Average model risk score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        _render_section_title("Risk Distribution")
        risk_counts = demo_df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Category", "Count"]
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                risk_counts,
                names="Risk_Category",
                values="Count",
                hole=0.62,
                color="Risk_Category",
                color_discrete_map=RISK_COLORS,
            )
            _apply_plotly_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(risk_counts.set_index("Risk_Category")["Count"])
        st.caption("This view shows how the AI system segments claims into operational risk tiers.")

    with right:
        _render_section_title("Decision Action Distribution")
        action_counts = demo_df["Decision_Action"].value_counts().reset_index()
        action_counts.columns = ["Decision_Action", "Count"]
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                action_counts,
                names="Decision_Action",
                values="Count",
                hole=0.62,
                color="Decision_Action",
                color_discrete_map=DECISION_COLORS,
            )
            _apply_plotly_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(action_counts.set_index("Decision_Action")["Count"])
        st.caption("Operational routing actions show how the model output translates into workflow.")

    second_left, second_right = st.columns(2)
    with second_left:
        _render_section_title("Fraud Probability Distribution")
        if PLOTLY_AVAILABLE:
            fig = px.histogram(
                demo_df,
                x="Model_Fraud_Probability",
                nbins=20,
                color_discrete_sequence=[PALETTE["blue"]],
            )
            for value in thresholds.values():
                fig.add_vline(x=value, line_dash="dash", line_color=PALETTE["red"])
            _apply_plotly_layout(fig, height=320)
            fig.update_xaxes(title="Fraud Probability")
            fig.update_yaxes(title="Claims")
            st.plotly_chart(fig, use_container_width=True)
        else:
            hist = demo_df["Model_Fraud_Probability"]
            st.bar_chart(hist.value_counts(bins=12).sort_index())
        st.caption("Higher concentrations on the right indicate a larger suspicious-claims workload.")

    with second_right:
        _render_section_title("Top 10 Highest-Risk Claims")
        top_claims = demo_df.nlargest(10, "Model_Fraud_Probability").sort_values("Model_Fraud_Probability", ascending=True)[
            [
                "Claim_ID",
                "Model_Fraud_Probability",
            ]
        ]
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                top_claims,
                x="Model_Fraud_Probability",
                y="Claim_ID",
                orientation="h",
                color_discrete_sequence=[PALETTE["red"]],
            )
            _apply_plotly_layout(fig, height=320)
            fig.update_xaxes(title="Fraud Probability")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top_claims.set_index("Claim_ID")["Model_Fraud_Probability"])

    _render_section_title("Top 10 Highest-Risk Cases")
    _display_table(
        demo_df.nlargest(10, "Model_Fraud_Probability"),
        columns_map={
            "Claim_ID": "Claim ID",
            "Provider_ID": "Provider ID",
            "Claim_Amount": "Claim Amount",
            "Model_Fraud_Probability": "Fraud Probability",
            "Risk_Category": "Risk Category",
            "Decision_Action": "Decision Action",
            "Top_Risk_Driver": "Top Risk Driver",
        },
        money_columns=["Claim_Amount"],
        probability_columns=["Model_Fraud_Probability"],
        max_rows=6 if presentation_mode else 10,
    )

elif page == "Batch Scoring":
    _render_header(
        "Batch Fraud Scoring",
        "Upload a claims file for scoring and keep it loaded across the dashboard. The app uses pre-scored probabilities if they already exist and only calls the model when raw inputs need scoring.",
    )

    _render_subsection_title("Step 1: Upload Data")
    uploaded_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"])
    data_source = st.session_state["current_data_source"]
    scoring_result = demo_df.copy() if not demo_df.empty else pd.DataFrame()
    persist_loaded_data = False

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        data_source = "uploaded file"
        if "Model_Fraud_Probability" in upload_df.columns:
            scoring_result = _apply_engine_safe(upload_df, thresholds)
            st.success("Uploaded CSV already contains model scores, so the dashboard is using them directly.")
            persist_loaded_data = True
        else:
            missing_columns = get_missing_synthetic_scoring_columns(upload_df)
            with st.expander("Scoring diagnostics", expanded=False):
                st.write("Columns not present from the synthetic model training schema:")
                st.write(missing_columns if missing_columns else "No missing model columns detected.")

            if st.button("Run scoring on uploaded claims"):
                try:
                    scoring_result = _apply_engine_safe(
                        score_synthetic_claims_with_context(upload_df),
                        thresholds,
                    )
                    st.success("Claims scored successfully using the saved synthetic demo model.")
                    persist_loaded_data = True
                except Exception as exc:
                    st.warning(
                        "The file could not be scored automatically. Please check whether the upload contains claim-level fields similar to the synthetic demo dataset."
                    )
                    st.code(str(exc))
                    scoring_result = _standardize_dashboard_frame(upload_df)
            else:
                st.info("Click 'Run scoring on uploaded claims' to score this file with the saved model.")
                scoring_result = _standardize_dashboard_frame(upload_df)

    if persist_loaded_data and not scoring_result.empty and "Model_Fraud_Probability" in scoring_result.columns:
        st.session_state["loaded_dataset"] = scoring_result.copy()
        st.session_state["current_data_source"] = "Uploaded Data Loaded"
        st.session_state["current_record_count"] = len(scoring_result)
        st.session_state["current_human_reviews"] = int(scoring_result.get("Human_Review_Required", pd.Series(dtype=bool)).sum())
        st.session_state["current_payment_holds"] = int(
            (scoring_result.get("Decision_Action", pd.Series(dtype=str)) == "Payment Hold + Mandatory Investigator Review").sum()
        )
        demo_df = scoring_result.copy()
    elif st.session_state["loaded_dataset"].empty:
        st.session_state["loaded_dataset"] = pd.DataFrame()
        st.session_state["current_data_source"] = "No Data Loaded"
        st.session_state["current_record_count"] = 0
        st.session_state["current_human_reviews"] = 0
        st.session_state["current_payment_holds"] = 0

    st.markdown(f'<div class="note-box">Current data source: <strong>{data_source}</strong>.</div>', unsafe_allow_html=True)

    _render_subsection_title("Step 2: Apply Filters")
    risk_options = ["High Risk", "Medium Risk", "Low Risk"]
    selected_risks = st.multiselect("Risk category filter", risk_options, default=risk_options)
    threshold = st.slider("Probability threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    top_n = st.selectbox("Top N records to show", [10, 20, 50, 100], index=0 if presentation_mode else 1)
    region_options = sorted(scoring_result["Region"].dropna().astype(str).unique().tolist()) if "Region" in scoring_result.columns else []
    claim_type_options = sorted(scoring_result["Claim_Type"].dropna().astype(str).unique().tolist()) if "Claim_Type" in scoring_result.columns else []
    selected_regions = region_options
    selected_claim_types = claim_type_options
    if not presentation_mode:
        with st.expander("Advanced Filters"):
            selected_regions = st.multiselect("Region filter", region_options, default=region_options)
            selected_claim_types = st.multiselect("Claim type filter", claim_type_options, default=claim_type_options)

    if not scoring_result.empty and "Risk_Category" in scoring_result.columns:
        filtered = _filtered_batch_frame(
            scoring_result,
            selected_risks=selected_risks,
            threshold=threshold,
            top_n=int(top_n),
            selected_regions=selected_regions,
            selected_claim_types=selected_claim_types,
        )
    else:
        filtered = pd.DataFrame()

    _render_subsection_title("Step 3: Review Scored Results")
    if filtered.empty:
        st.info("Upload a CSV to see scored results here.")
    else:
        _display_table(
            filtered,
            columns_map={
                "Claim_ID": "Claim ID",
                "Provider_ID": "Provider ID",
                "Claim_Amount": "Claim Amount",
                "Model_Fraud_Probability": "Fraud Probability",
                "Risk_Category": "Risk Category",
                "Decision_Action": "Decision Action",
                "Human_Review_Required": "Human Review",
                "Top_Risk_Driver": "Top Risk Driver",
            },
            money_columns=["Claim_Amount"],
            probability_columns=["Model_Fraud_Probability"],
            bool_columns=["Human_Review_Required"],
            max_rows=int(top_n),
        )
        st.caption("This top table is sorted by fraud probability for quick review, but the full loaded dataset is available below.")

    _render_subsection_title("Step 4: Download Scored CSV")
    if not filtered.empty:
        st.download_button(
            "Download scored CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="batch_scored_claims.csv",
            mime="text/csv",
        )

    with st.expander("View filtered results with all matching rows", expanded=False):
        if filtered.empty:
            st.info("No filtered dataset available yet.")
        else:
            st.dataframe(
                filtered[
                    [
                        column
                        for column in [
                            "Claim_ID",
                            "Provider_ID",
                            "Patient_ID",
                            "Claim_Type",
                            "Region",
                            "Provider_Type",
                            "Claim_Amount",
                            "Reimbursed_Amount",
                            "Deductible_Amount",
                            "Model_Fraud_Probability",
                            "Risk_Category",
                            "Automation_Risk_Tier",
                            "Decision_Action",
                            "Top_Risk_Driver",
                            "Secondary_Risk_Score",
                            "Secondary_Check_Reason",
                            "Human_Review_Required",
                        ]
                        if column in filtered.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View full loaded dataset with all columns", expanded=False):
        if scoring_result.empty:
            st.info("No scored dataset loaded yet.")
        else:
            st.dataframe(scoring_result, use_container_width=True, hide_index=True)

elif page == "Decision Engine":
    _render_header(
        "Decision Automation Engine",
        "This page shows how model fraud probabilities are converted into operational decisions using editable thresholds and secondary rules for borderline claims.",
    )
    if demo_df.empty:
        _empty_dashboard_message("Decision Engine")
        st.stop()

    _render_subsection_title("A. Threshold Controls")
    control_left, control_right = st.columns([1.1, 1.2])
    with control_left:
        low_upper = st.slider("Low risk upper threshold", 0.05, 0.80, float(thresholds["low_upper"]), 0.01, key="low_upper")
        borderline_upper = st.slider("Borderline risk upper threshold", 0.10, 0.85, float(thresholds["borderline_upper"]), 0.01, key="borderline_upper")
        elevated_upper = st.slider("Elevated risk upper threshold", 0.20, 0.95, float(thresholds["elevated_upper"]), 0.01, key="elevated_upper")
        high_upper = st.slider("High risk upper threshold", 0.30, 0.99, float(thresholds["high_upper"]), 0.01, key="high_upper")
        if st.button("Reset to Default Thresholds"):
            _reset_thresholds()
            st.rerun()

    thresholds = _current_thresholds()
    routed_df = _apply_engine_safe(demo_df, thresholds)

    with control_right:
        st.markdown(
            '<div class="note-box">Default routing: low risk claims can be auto approved, borderline claims go through secondary rules, elevated and high-risk claims go to human review, and critical-risk claims move to payment hold plus mandatory investigator review.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-card">
                <strong>Threshold guide</strong><br>
                Low Risk &lt; 0.40<br>
                Borderline 0.40-0.60<br>
                Elevated 0.60-0.70<br>
                High 0.70-0.90<br>
                Critical &gt; 0.90
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not thresholds_are_valid(thresholds):
            st.warning("Please keep the thresholds in increasing order: low < borderline < elevated < high.")
        else:
            st.success("Threshold order is valid.")

    _render_subsection_title("B. Routing Summary")
    action_summary = summarize_decision_actions(routed_df)
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    with a1:
        _render_kpi("Auto Approved", f'{action_summary["Auto Approve"]:,}')
    with a2:
        _render_kpi("Secondary Rules Checked", f'{action_summary["Secondary Rules Checked"]:,}')
    with a3:
        _render_kpi("Human Review Queue", f'{action_summary["Human Review Queue"]:,}')
    with a4:
        _render_kpi("Priority Investigations", f'{action_summary["Priority Investigation"]:,}')
    with a5:
        _render_kpi("Payment Holds", f'{action_summary["Payment Holds"]:,}')
    with a6:
        _render_kpi("Total Human Reviews Required", f'{action_summary["Total Human Reviews Required"]:,}')

    _render_subsection_title("C. Decision Visuals")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        _render_section_title("Decision Action Distribution")
        action_counts = routed_df["Decision_Action"].value_counts().reset_index()
        action_counts.columns = ["Decision_Action", "Count"]
        if PLOTLY_AVAILABLE:
            fig = px.pie(action_counts, names="Decision_Action", values="Count", hole=0.58)
            fig.update_traces(marker=dict(colors=[DECISION_COLORS.get(name, PALETTE["blue"]) for name in action_counts["Decision_Action"]]))
            _apply_plotly_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(action_counts.set_index("Decision_Action")["Count"])

    with chart_right:
        _render_section_title("Automation Risk Tier Counts")
        tier_counts = routed_df["Automation_Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Automation_Risk_Tier", "Count"]
        _render_plotly_or_bar(tier_counts, x="Automation_Risk_Tier", y="Count", color="Automation_Risk_Tier", title="Automation Risk Tiers")

    chart_low, chart_high = st.columns(2)
    with chart_low:
        _render_section_title("Fraud Probability Distribution")
        if PLOTLY_AVAILABLE:
            fig = px.histogram(
                routed_df,
                x="Model_Fraud_Probability",
                nbins=24,
                color_discrete_sequence=[PALETTE["blue"]],
            )
            for value in thresholds.values():
                fig.add_vline(x=value, line_dash="dash", line_color=PALETTE["red"])
            fig.update_layout(
                plot_bgcolor="#1a1a24",
                paper_bgcolor="#1a1a24",
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(color=PALETTE["navy"]),
            )
            fig.update_xaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
            fig.update_yaxes(color=PALETTE["muted"], gridcolor=PALETTE["border"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(routed_df["Model_Fraud_Probability"].value_counts(bins=20).sort_index())

    with chart_high:
        _render_section_title("Average Claim Amount by Decision Action")
        average_claim_amount = (
            routed_df.groupby("Decision_Action", as_index=False)["Claim_Amount"].mean().sort_values("Claim_Amount", ascending=False)
        )
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                average_claim_amount,
                x="Claim_Amount",
                y="Decision_Action",
                orientation="h",
                color="Decision_Action",
                color_discrete_map=DECISION_COLORS,
            )
            _apply_plotly_layout(fig, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(average_claim_amount.set_index("Decision_Action")["Claim_Amount"])

    _render_subsection_title("D. Routed Claims")
    decision_options = sorted(routed_df["Decision_Action"].dropna().unique().tolist())
    selected_actions = decision_options
    tier_options = sorted(routed_df["Automation_Risk_Tier"].dropna().unique().tolist())
    selected_tiers = tier_options
    review_filter = "All"
    min_probability = 0.0
    with st.expander("Filter routed claims", expanded=not presentation_mode):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            selected_actions = st.multiselect("Decision action filter", decision_options, default=decision_options)
        with filter_col2:
            selected_tiers = st.multiselect("Risk tier filter", tier_options, default=tier_options)
        with filter_col3:
            review_filter = st.selectbox("Human review required", ["All", "True", "False"])
        with filter_col4:
            min_probability = st.slider("Minimum fraud probability", 0.0, 1.0, 0.0, 0.01, key="decision_min_probability")

    filtered_routed = routed_df.copy()
    filtered_routed = filtered_routed[filtered_routed["Decision_Action"].isin(selected_actions)]
    filtered_routed = filtered_routed[filtered_routed["Automation_Risk_Tier"].isin(selected_tiers)]
    filtered_routed = filtered_routed[filtered_routed["Model_Fraud_Probability"] >= min_probability]
    if review_filter != "All":
        filtered_routed = filtered_routed[filtered_routed["Human_Review_Required"] == (review_filter == "True")]

    _render_section_title("Routed Claims")
    _display_table(
        filtered_routed,
        columns_map={
            "Claim_ID": "Claim ID",
            "Provider_ID": "Provider ID",
            "Claim_Amount": "Claim Amount",
            "Model_Fraud_Probability": "Fraud Probability",
            "Automation_Risk_Tier": "Automation Tier",
            "Secondary_Risk_Score": "Secondary Score",
            "Decision_Action": "Decision Action",
            "Human_Review_Required": "Human Review",
            "Secondary_Check_Reason": "Secondary Check Reason",
        },
        money_columns=["Claim_Amount"],
        probability_columns=["Model_Fraud_Probability"],
        bool_columns=["Human_Review_Required"],
        max_rows=8 if presentation_mode else 20,
    )
    st.caption("This summary table is trimmed for readability. Full routed rows and all columns are available below.")

    with st.expander("View full routed dataset with all columns", expanded=False):
        st.dataframe(filtered_routed, use_container_width=True, hide_index=True)

    st.download_button(
        "Download routed claims CSV",
        data=filtered_routed.to_csv(index=False).encode("utf-8"),
        file_name="decision_engine_routed_claims.csv",
        mime="text/csv",
    )

elif page == "Case Review":
    _render_header(
        "High-Risk Case Review",
        "Investigator view for explaining why a claim was flagged and what action should happen next. This page is designed for a fraud-operations walkthrough rather than a technical model demo.",
    )
    if demo_df.empty:
        _empty_dashboard_message("Case Review")
        st.stop()

    high_risk_df = demo_df[
        demo_df["Automation_Risk_Tier"].isin(["High Risk", "Critical Risk", "Elevated Risk"])
    ].copy()
    if high_risk_df.empty:
        st.warning("No high-risk cases are available in the loaded dataset.")
    else:
        selected_claim_id = st.selectbox("Select a high-risk claim", high_risk_df["Claim_ID"].tolist())
        case_row = high_risk_df.loc[high_risk_df["Claim_ID"] == selected_claim_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _render_kpi("Claim ID", str(case_row["Claim_ID"]))
        with c2:
            _render_kpi("Provider ID", str(case_row["Provider_ID"]))
        with c3:
            _render_kpi("Fraud Probability", f'{case_row["Model_Fraud_Probability"]:.1%}')
        with c4:
            _render_kpi("Claim Amount", f'${case_row["Claim_Amount"]:,.0f}')

        _render_subsection_title("Case Routing Summary")
        _render_badge_row(
            [
                (str(case_row["Risk_Category"]), "risk"),
                (str(case_row["Automation_Risk_Tier"]), "risk"),
                (str(case_row["Decision_Action"]), "decision"),
                ("Human Review Required" if bool(case_row["Human_Review_Required"]) else "No Immediate Review", "decision"),
            ]
        )

        _render_section_title("Risk Display")
        probability_value = float(case_row["Model_Fraud_Probability"])
        if PLOTLY_AVAILABLE:
            gauge_color = PALETTE["green"] if probability_value < 0.40 else PALETTE["amber"] if probability_value < 0.70 else PALETTE["orange"] if probability_value < 0.90 else PALETTE["critical"]
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_value * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 40], "color": "#14532d"},
                            {"range": [40, 70], "color": "#78350f"},
                            {"range": [70, 90], "color": "#7c2d12"},
                            {"range": [90, 100], "color": "#4c0519"},
                        ],
                    },
                )
            )
            _apply_plotly_layout(fig, height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.progress(probability_value)
        st.caption("A higher indicator means the model sees stronger fraud-like patterns in this claim.")

        left, right = st.columns([1.0, 1.15])
        with left:
            _render_section_title("Reason For Flag")
            bullets = [
                f"Top risk driver: {case_row['Top_Risk_Driver']}",
                f"Claim type: {case_row['Claim_Type']}",
                f"Region: {case_row['Region']}",
                f"Provider type: {case_row['Provider_Type']}",
                f"Prior fraud flag: {_format_yes_no(case_row['Prior_Fraud_Flag'])}",
                f"Secondary risk score: {case_row.get('Secondary_Risk_Score', 0)}",
                f"Secondary check reason: {case_row.get('Secondary_Check_Reason', 'No secondary check required')}",
            ]
            for bullet in bullets:
                st.markdown(f'<div class="insight-card" style="margin-bottom:0.5rem;padding:0.7rem 0.85rem;">{bullet}</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="note-box">This system routes claims for review. It does not automatically deny healthcare claims.</div>',
                unsafe_allow_html=True,
            )

        with right:
            _render_section_title("Potential Contributing Features")
            _render_case_feature_chart(case_row)

        _render_section_title("Similar High-Risk Cases")
        similar_cases = high_risk_df[
            (high_risk_df["Provider_ID"] == case_row["Provider_ID"])
            | (high_risk_df["Top_Risk_Driver"] == case_row["Top_Risk_Driver"])
        ].head(8)
        _display_table(
            similar_cases,
            columns_map={
                "Claim_ID": "Claim ID",
                "Provider_ID": "Provider ID",
                "Claim_Amount": "Claim Amount",
                "Model_Fraud_Probability": "Fraud Probability",
                "Top_Risk_Driver": "Top Risk Driver",
                "Decision_Action": "Decision Action",
            },
            money_columns=["Claim_Amount"],
            probability_columns=["Model_Fraud_Probability"],
            max_rows=5,
        )
        if not presentation_mode:
            with st.expander("View full similar cases table"):
                st.dataframe(similar_cases, use_container_width=True, hide_index=True)

elif page == "Model Insights":
    _render_header(
        "Model Insights",
        "Technical view for evaluating fraud detection performance. This page is designed to show that accuracy alone is not enough in an imbalanced healthcare fraud setting.",
    )

    if provider_leaderboard.empty:
        st.warning("Real metrics files were not found. Showing demo metrics for presentation layout.")
        provider_leaderboard = _placeholder_model_table()

    best_row = provider_leaderboard.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        _render_kpi("Best Model", str(best_row["model"]))
    with m2:
        _render_kpi("ROC-AUC", f'{best_row["roc_auc"]:.3f}')
    with m3:
        _render_kpi("Recall", f'{best_row["recall"]:.3f}')
    with m4:
        _render_kpi("Precision", f'{best_row["precision"]:.3f}')
    with m5:
        _render_kpi("F1 Score", f'{best_row["f1"]:.3f}')

    _render_section_title("Primary Model Comparison")
    st.dataframe(provider_leaderboard, use_container_width=True)
    st.caption(
        "Fraud detection should not optimize accuracy alone because missing fraud and incorrectly flagging legitimate claims have different costs."
    )

    if PLOTLY_AVAILABLE:
        plot_df = provider_leaderboard.melt(
            id_vars="model",
            value_vars=["precision", "recall", "f1", "roc_auc"],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(plot_df, x="Metric", y="Score", color="model", barmode="group")
        _apply_plotly_layout(fig, title="Model Comparison Across Key Metrics", height=340)
        st.plotly_chart(fig, use_container_width=True)

    if synthetic_leaderboard.empty:
        synthetic_leaderboard = _placeholder_model_table().rename(columns={"model": "synthetic_model"})

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        _render_section_title("Confusion Matrix")
        cm_path = CHARTS_DIR / "provider_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Primary provider model confusion matrix", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

    with r1c2:
        _render_section_title("Precision-Recall Curve")
        pr_path = CHARTS_DIR / "provider_precision_recall_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Primary provider model precision-recall curve", use_container_width=True)
        else:
            st.info("Precision-recall image not found.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        _render_section_title("ROC Curve")
        roc_path = CHARTS_DIR / "provider_roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Primary provider model ROC curve", use_container_width=True)
        else:
            st.info("ROC curve image not found.")

    with r2c2:
        _render_section_title("Feature Importance")
        fi_path = CHARTS_DIR / "provider_feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), caption="Tree-based feature importance for explanation", use_container_width=True)
        elif not provider_importance.empty:
            _render_plotly_or_bar(provider_importance.head(12), x="feature", y="importance", title="Feature Importance")
        else:
            st.info("Feature importance output not found.")

    with st.expander("View threshold details", expanded=not presentation_mode):
        _render_section_title("Threshold Comparison")
        if provider_thresholds.empty:
            st.info("Threshold metrics file not found. Showing presentation-friendly placeholder values.")
            provider_thresholds = pd.DataFrame(
                [
                    {"threshold": 0.30, "precision": 0.32, "recall": 0.95, "f1": 0.48, "roc_auc": 0.95},
                    {"threshold": 0.50, "precision": 0.44, "recall": 0.87, "f1": 0.59, "roc_auc": 0.95},
                    {"threshold": 0.70, "precision": 0.54, "recall": 0.76, "f1": 0.64, "roc_auc": 0.95},
                ]
            )
        st.dataframe(provider_thresholds, use_container_width=True, hide_index=True)

    with st.expander("View raw model metrics", expanded=False):
        st.write("Primary model leaderboard")
        st.dataframe(provider_leaderboard, use_container_width=True, hide_index=True)
        st.write("Secondary synthetic model comparison")
        st.dataframe(synthetic_leaderboard, use_container_width=True, hide_index=True)

elif page == "Business Impact":
    _render_header(
        "Business Impact",
        "Translate model performance into operational savings. These sliders are designed for live classroom discussion about fraud prevention value versus review cost.",
    )

    left, right = st.columns([0.95, 1.05])
    with left:
        avg_fraud_claim_amount = st.slider("Average fraudulent claim amount", 1000, 30000, 12000, step=500)
        monthly_claims = st.slider("Claims/providers reviewed per month", 500, 25000, 5000, step=500)
        estimated_fraud_rate = st.slider("Estimated fraud rate", 0.01, 0.20, 0.06, step=0.01)
        manual_review_cost = st.slider("Manual review cost per claim", 10, 250, 65, step=5)
        percent_fraud_caught = st.slider("Percentage of fraud caught by model", 0.10, 1.0, 0.72, step=0.01)
        percent_sent_to_review = st.slider("Percentage of claims sent to human review", 0.01, 0.75, 0.18, step=0.01)

    estimated_fraud_cases = monthly_claims * estimated_fraud_rate
    fraud_caught = estimated_fraud_cases * percent_fraud_caught
    monthly_fraud_prevented = fraud_caught * avg_fraud_claim_amount
    monthly_review_cost = monthly_claims * percent_sent_to_review * manual_review_cost
    monthly_net_savings = monthly_fraud_prevented - monthly_review_cost
    annual_net_savings = monthly_net_savings * 12

    with right:
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-bottom:0.8rem;">
                <div class="kpi-card">
                    <div class="kpi-label">Monthly Fraud Prevented</div>
                    <div class="kpi-value">{_format_currency(monthly_fraud_prevented, compact=True)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Annual Fraud Prevented</div>
                    <div class="kpi-value">{_format_currency(monthly_fraud_prevented * 12, compact=True)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Monthly Review Cost</div>
                    <div class="kpi-value">{_format_currency(monthly_review_cost, compact=True)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Estimated Net Savings</div>
                    <div class="kpi-value">{_format_currency(annual_net_savings, compact=True)}</div>
                    <div class="kpi-sub">Annualized estimate</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        compare_df = pd.DataFrame(
            {
                "Category": ["Fraud Prevented", "Review Cost", "Net Savings"],
                "Amount": [monthly_fraud_prevented, monthly_review_cost, monthly_net_savings],
            }
        )
        if PLOTLY_AVAILABLE:
            fig = px.bar(compare_df, x="Category", y="Amount", color="Category", color_discrete_sequence=[PALETTE["green"], PALETTE["amber"], PALETTE["blue"]])
            _apply_plotly_layout(fig, title="Monthly Value vs Cost", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _render_plotly_or_bar(compare_df, x="Category", y="Amount", title="Monthly Value vs Cost")

        annual_df = pd.DataFrame({"Metric": ["Monthly", "Annual"], "Amount": [monthly_net_savings, annual_net_savings]})
        _render_plotly_or_bar(annual_df, x="Metric", y="Amount", title="Monthly vs Annual Impact")

    st.markdown(
        '<div class="note-box">These estimates are adjustable assumptions for business planning, not guaranteed financial outcomes. The model creates value when it prevents high-cost suspicious payments without overwhelming investigators with unnecessary reviews.</div>',
        unsafe_allow_html=True,
    )

elif page == "Responsible AI":
    _render_header(
        "Responsible AI Controls",
        "Healthcare fraud detection has real operational and fairness consequences. This page explains why the model is designed as a decision-support tool, not a fully autonomous claims decision-maker.",
    )

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:1.2rem;">
        <div class="insight-card">
            <div class="section-title" style="margin-bottom:0.6rem;">False Positive Cost</div>
            A legitimate claim may be flagged by mistake. This can delay payment, frustrate providers and patients,
            increase administrative burden, and in some cases delay care delivery.
        </div>
        <div class="insight-card" style="grid-row: span 2;">
            <div class="section-title" style="margin-bottom:0.6rem;">Human-In-The-Loop Rules</div>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.9;color:inherit;">
                <li>The system does not automatically deny healthcare claims.</li>
                <li>Critical-risk claims are placed on payment hold and routed to mandatory human review.</li>
                <li>Borderline claims go through a secondary rule-based check.</li>
                <li>Claims above $10,000 require human sign-off before adverse action.</li>
                <li>Model confidence below 80% requires secondary review.</li>
                <li>Providers with no prior fraud history cannot be routed to adverse action without investigator review.</li>
                <li>Appeals must always be reviewed by a human investigator.</li>
                <li>Human investigators make final denial decisions.</li>
                <li>The model should assist investigators, not replace them.</li>
            </ul>
        </div>
        <div class="insight-card">
            <div class="section-title" style="margin-bottom:0.6rem;">False Negative Cost</div>
            A fraudulent claim may be approved and paid. This creates direct financial loss and allows abusive billing
            patterns to continue.
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-top:1.2rem;">
        <div class="insight-card">
            <div class="section-title" style="margin-bottom:0.6rem;">Bias And Fairness Risks</div>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.9;color:inherit;">
                <li>Over-flagging underserved regions</li>
                <li>Penalizing smaller clinics with unusual billing patterns</li>
                <li>Penalizing complex care cases</li>
                <li>Delaying care for vulnerable patient populations</li>
            </ul>
        </div>
        <div class="insight-card">
            <div class="section-title" style="margin-bottom:0.6rem;">Mitigation Controls</div>
            <ul style="margin:0;padding-left:1.2rem;line-height:1.9;color:inherit;">
                <li>Threshold tuning based on business cost tradeoffs</li>
                <li>Bias audits by region and provider type</li>
                <li>Human review before payment action</li>
                <li>Appeal workflow and case escalation</li>
                <li>Ongoing monitoring and periodic retraining</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "About":
    _render_header(
        "About Project",
        "Summary page for the professor walkthrough: business problem, users, datasets, AI approach, selected model, limitations, and future roadmap.",
    )

    best_model = _load_json(str(METRICS_DIR / "training_summary.json")).get("provider_model", "logistic_regression")
    cards = [
        ("Business Problem", "Manual review and static rules can miss suspicious healthcare claims while creating false positives and slow investigations."),
        ("Customer / User", "Small and mid-sized health insurers, Medicaid administrators, and third-party claims processors."),
        ("Datasets Used", "Healthcare Provider Fraud Detection Analysis; Health Insurance Claims Data for Fraud Detection; CMS Medicare data context."),
        ("AI Approach", "Classification and prediction for provider-level and claim-level fraud risk scoring."),
        ("Models Tested", "Logistic Regression, Random Forest, and XGBoost or sklearn gradient boosting fallback."),
        ("Best Model Selected", str(best_model)),
        ("Limitations", "The primary dataset predicts potential fraud rather than final legal determination, and the synthetic dataset mainly supports demo software and comparison."),
        ("Future Improvements", "Add investigator feedback loops, bias audits, network analytics, calibration, and richer local explainability."),
    ]
    cards_html = "".join(
        f'''<div class="insight-card" style="break-inside:avoid;">
            <div class="subsection-title">{title}</div>
            <div style="color:{PALETTE["muted"]};font-size:0.9rem;line-height:1.6;">{body}</div>
        </div>'''
        for title, body in cards
    )
    st.markdown(
        f'''<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">{cards_html}</div>''',
        unsafe_allow_html=True,
    )