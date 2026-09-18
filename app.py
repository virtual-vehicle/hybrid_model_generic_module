# Copyright (c) 2026 Virtual Vehicle Research GmbH
# SPDX-License-Identifier: MIT
#
# Part of the Hybrid Model Generic Module developed within the SoliDAIR project.
# See LICENSE.md in the project root for the full license terms.

import ast
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from regression_training import (
    ANN_INSTALL_GUIDANCE,
    ANN_MODEL_NAME,
    AVAILABLE_REGRESSION_MODELS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    get_numeric_model_columns,
    is_ann_available,
    train_and_evaluate_regression_models,
)
from regression_training import (
    RANDOM_STATE as MODEL_RANDOM_STATE,
)

MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = [".csv", ".txt", ".tsv", ".parquet"]
STEP_LABELS = {
    1: "Upload Datasets",
    2: "Datasets' preprocessing",
    3: "Imbalance Analysis",
    4: "Create the Hybrid Dataset",
    5: "Analyze the Hybrid Dataset",
    6: "Train Regression Models",
    7: "Summary & Downloads",
}
DEFAULT_IMBALANCE_THRESHOLDS = {
    "input_1": {"lower": 0.553699, "upper": 0.649165},
    "input_2": {"lower": 0.480418, "upper": 0.584856},
    "input_3": {"lower": 0.017045, "upper": 0.147727},
    "input_4": {"lower": 0.058394, "upper": 0.175182},
    "input_5": {"lower": 0.338843, "upper": 0.553719},
}
MAX_WITHIN_THRESHOLD_PLOT_POINTS = 2_000
MAX_PRODUCTION_PLOT_ROWS = 2_000
MAX_SIMULATION_PLOT_ROWS = 2_000
MAX_PREDICTION_PLOT_ROWS = 10_000
MAX_TRUE_NEGATIVE_PREDICTION_PLOT_ROWS = 2_000
MAX_PRIORITY_PREDICTION_PLOT_ROWS = 20_000
MAX_FEATURE_SELECTION_ROWS = 10_000
FEATURE_SELECTION_RF_ESTIMATORS = 100
RANDOM_STATE = 42
USER_FACING_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    OSError,
    UnicodeError,
    ImportError,
    RuntimeError,
    pd.errors.ParserError,
)
ACTION_GREEN = "#2e7d32"
ACTION_GREEN_HOVER = "#256628"
ACTION_GREEN_BORDER = "#1b5e20"
ACTION_GREEN_SOFT = "rgba(46, 125, 50, 0.12)"
CHECKBOX_BLUE = "#1565c0"


@dataclass
class SchemaCheckResult:
    common_columns: List[str]
    only_in_experimental: List[str]
    only_in_simulation: List[str]
    dtype_mismatches: List[Dict[str, str]]

    @property
    def has_column_mismatch(self) -> bool:
        return bool(self.only_in_experimental or self.only_in_simulation)

    @property
    def has_dtype_mismatch(self) -> bool:
        return bool(self.dtype_mismatches)

    @property
    def all_match(self) -> bool:
        return not self.has_column_mismatch and not self.has_dtype_mismatch


def init_state() -> None:
    defaults = {
        "experimental_df": None,
        "simulation_df": None,
        "schema_check": None,
        "merged_df": None,
        "merge_details": None,
        "hybrid_analysis_completed": False,
        "model_metrics_df": None,
        "model_predictions_df": None,
        "training_details": None,
        "trained_models": None,
        "imbalance_result_df": None,
        "imbalance_ratio": None,
        "imbalance_column": None,
        "imbalance_thresholds": None,
        "imbalance_summary": None,
        "feature_selection_result_df": None,
        "feature_selection_summary": None,
        "feature_drop_success_message": None,
        "reset_feature_drop_checkbox": False,
        "feature_selection_success_message": None,
        "reset_feature_selection_checkbox": False,
        "reset_column_mapping_checkbox": False,
        "hybrid_balance_success_message": None,
        "auto_run_schema_check": False,
        "column_mapping_applied": False,
        "applied_column_mapping": {},
        "current_step": 1,
        "scroll_to_top": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_downstream_state() -> None:
    st.session_state["schema_check"] = None
    st.session_state["merged_df"] = None
    st.session_state["merge_details"] = None
    st.session_state["hybrid_analysis_completed"] = False
    st.session_state["model_metrics_df"] = None
    st.session_state["model_predictions_df"] = None
    st.session_state["training_details"] = None
    st.session_state["trained_models"] = None
    st.session_state["imbalance_result_df"] = None
    st.session_state["imbalance_ratio"] = None
    st.session_state["imbalance_column"] = None
    st.session_state["imbalance_thresholds"] = None
    st.session_state["imbalance_summary"] = None
    st.session_state["feature_selection_result_df"] = None
    st.session_state["feature_selection_summary"] = None
    st.session_state["feature_drop_success_message"] = None
    st.session_state["reset_feature_drop_checkbox"] = False
    st.session_state["feature_selection_success_message"] = None
    st.session_state["reset_feature_selection_checkbox"] = False
    st.session_state["reset_column_mapping_checkbox"] = False
    st.session_state["hybrid_balance_success_message"] = None
    st.session_state["auto_run_schema_check"] = False
    st.session_state["column_mapping_applied"] = False
    st.session_state["applied_column_mapping"] = {}


def reset_post_schema_state() -> None:
    st.session_state["merged_df"] = None
    st.session_state["merge_details"] = None
    st.session_state["hybrid_analysis_completed"] = False
    st.session_state["model_metrics_df"] = None
    st.session_state["model_predictions_df"] = None
    st.session_state["training_details"] = None
    st.session_state["trained_models"] = None
    st.session_state["imbalance_result_df"] = None
    st.session_state["imbalance_ratio"] = None
    st.session_state["imbalance_column"] = None
    st.session_state["imbalance_thresholds"] = None
    st.session_state["imbalance_summary"] = None
    st.session_state["feature_selection_result_df"] = None
    st.session_state["feature_selection_summary"] = None
    st.session_state["feature_selection_success_message"] = None
    st.session_state["reset_feature_selection_checkbox"] = False
    st.session_state["reset_column_mapping_checkbox"] = False
    st.session_state["hybrid_balance_success_message"] = None


def go_to_step(step_number: int) -> None:
    st.session_state["current_step"] = step_number
    st.session_state["scroll_to_top"] = True
    st.rerun()


def scroll_to_top_once() -> None:
    if not st.session_state.get("scroll_to_top"):
        return

    st.session_state["scroll_to_top"] = False
    st.iframe(
        """
        <script>
            const scrollTargets = [
                window.parent,
                window.parent.document.documentElement,
                window.parent.document.body,
                window.parent.document.querySelector('.main'),
                window.parent.document.querySelector('[data-testid="stAppViewContainer"]'),
                window.parent.document.querySelector('[data-testid="stMain"]')
            ];
            scrollTargets.forEach((target) => {
                if (!target) return;
                if (typeof target.scrollTo === 'function') {
                    target.scrollTo({ top: 0, left: 0, behavior: 'instant' });
                } else {
                    target.scrollTop = 0;
                }
            });
        </script>
        """,
        height=1,
        width=1,
        tab_index=-1,
    )


def apply_app_styles() -> None:
    st.markdown(
        f"""
        <style>
            div[data-testid="stButton"] > button {{
                background-color: {ACTION_GREEN};
                border-color: {ACTION_GREEN_BORDER};
                color: #ffffff;
            }}

            div[data-testid="stButton"] > button:hover:enabled {{
                background-color: {ACTION_GREEN_HOVER};
                border-color: {ACTION_GREEN_BORDER};
                color: #ffffff;
            }}

            div[data-testid="stButton"] > button:active:enabled,
            div[data-testid="stButton"] > button:focus:enabled {{
                background-color: transparent;
                border-color: {ACTION_GREEN};
                color: {ACTION_GREEN};
            }}

            div[data-testid="stButton"] > button:disabled {{
                background-color: rgba(46, 125, 50, 0.28);
                border-color: rgba(46, 125, 50, 0.28);
                color: rgba(255, 255, 255, 0.72);
            }}

            div[data-testid="stCheckbox"] {{
                --primary-color: {CHECKBOX_BLUE};
            }}

            div[data-testid="stCheckbox"] input[type="checkbox"] {{
                accent-color: {CHECKBOX_BLUE};
            }}

            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child,
            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child > div,
            div[data-testid="stCheckbox"] label:has(input[type="checkbox"]:checked) span:first-child,
            div[data-testid="stCheckbox"] label:has(input[type="checkbox"]:checked) span:first-child > div,
            div[data-testid="stCheckbox"] span[data-baseweb="checkbox"][aria-checked="true"],
            div[data-testid="stCheckbox"] span[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked),
            div[data-testid="stCheckbox"] span[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > div {{
                background-color: {CHECKBOX_BLUE} !important;
                border-color: {CHECKBOX_BLUE} !important;
            }}

            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child::before,
            div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child::after,
            div[data-testid="stCheckbox"] label:has(input[type="checkbox"]:checked) span:first-child::before,
            div[data-testid="stCheckbox"] label:has(input[type="checkbox"]:checked) span:first-child::after {{
                background-color: {CHECKBOX_BLUE} !important;
                border-color: {CHECKBOX_BLUE} !important;
            }}

            div[data-testid="stCheckbox"] label:has(input[type="checkbox"]:checked) svg,
            div[data-testid="stCheckbox"] span[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) svg {{
                color: #ffffff !important;
                fill: #ffffff !important;
            }}

            div[data-testid="stRadio"] [role="radio"][aria-checked="true"] div:first-child {{
                background-color: {ACTION_GREEN};
                border-color: {ACTION_GREEN};
            }}

            div[data-baseweb="tag"] {{
                background-color: {ACTION_GREEN_SOFT};
                color: {ACTION_GREEN_BORDER};
            }}

            div[data-baseweb="tag"] svg {{
                color: {ACTION_GREEN_BORDER};
            }}

            [role="option"][aria-selected="true"] {{
                background-color: {ACTION_GREEN_SOFT};
                color: {ACTION_GREEN_BORDER};
            }}

            .ant-steps-item-process .ant-steps-item-icon,
            .ant-steps-item-finish .ant-steps-item-icon {{
                background-color: {ACTION_GREEN} !important;
                border-color: {ACTION_GREEN} !important;
            }}

            .ant-steps-item-process .ant-steps-icon,
            .ant-steps-item-finish .ant-steps-icon,
            .ant-steps-item-finish .ant-steps-icon svg {{
                color: #ffffff !important;
            }}

            .ant-steps-item-finish .ant-steps-item-tail::after {{
                background-color: {ACTION_GREEN} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_unlocked_step() -> int:
    unlocked = 1
    if (
        st.session_state["experimental_df"] is not None
        and st.session_state["simulation_df"] is not None
    ):
        unlocked = 2
    if st.session_state["schema_check"] is not None:
        unlocked = 3
    if st.session_state["imbalance_result_df"] is not None:
        unlocked = 4
    if st.session_state["merged_df"] is not None:
        unlocked = 5
    if st.session_state["hybrid_analysis_completed"]:
        unlocked = 6
    if st.session_state["model_metrics_df"] is not None:
        unlocked = 7
    return unlocked


def render_dataset_preview(df_exp: pd.DataFrame, df_sim: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Real-world dataset preview:**")
        st.markdown(f"Shape: {df_exp.shape[0]} rows x {df_exp.shape[1]} columns")
        render_dataframe(df_exp.head(10), width="stretch")
    with c2:
        st.markdown("**Synthetic dataset preview:**")
        st.markdown(f"Shape: {df_sim.shape[0]} rows x {df_sim.shape[1]} columns")
        render_dataframe(df_sim.head(10), width="stretch")


def get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.lower().split(".")[-1]


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    ext = get_extension(uploaded_file.name)
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: `{ext}`."
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        return False, (
            f"File `{uploaded_file.name}` exceeds {MAX_FILE_SIZE_MB} MB "
            f"({uploaded_file.size / (1024 * 1024):.2f} MB)."
        )
    return True, ""


def load_dataframe(uploaded_file) -> pd.DataFrame:
    ext = get_extension(uploaded_file.name)
    raw = uploaded_file.getvalue()
    stream = io.BytesIO(raw)

    if ext == ".csv":
        return pd.read_csv(stream)
    if ext in [".txt", ".tsv"]:
        return pd.read_csv(stream, sep=None, engine="python")
    if ext == ".parquet":
        return pd.read_parquet(stream)
    raise ValueError(f"Unsupported extension: {ext}")


def make_arrow_compatible_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in display_df.columns:
        series = display_df[column]
        if series.dtype == "object":
            non_null_types = series.dropna().map(type).unique()
            if len(non_null_types) > 1:
                display_df[column] = series.astype("string")
    return display_df


def render_dataframe(data, **kwargs) -> None:
    kwargs.setdefault("width", "stretch")
    if isinstance(data, pd.DataFrame):
        data = make_arrow_compatible_dataframe(data)
    st.dataframe(data, **kwargs)


def check_schema(df_exp: pd.DataFrame, df_sim: pd.DataFrame) -> SchemaCheckResult:
    exp_cols = set(df_exp.columns)
    sim_cols = set(df_sim.columns)
    common_columns = sorted(exp_cols.intersection(sim_cols))
    only_in_experimental = sorted(exp_cols.difference(sim_cols))
    only_in_simulation = sorted(sim_cols.difference(exp_cols))

    mismatches = []
    for col in common_columns:
        exp_dtype = str(df_exp[col].dtype)
        sim_dtype = str(df_sim[col].dtype)
        if exp_dtype != sim_dtype:
            mismatches.append(
                {
                    "column": col,
                    "experimental_dtype": exp_dtype,
                    "simulation_dtype": sim_dtype,
                }
            )

    return SchemaCheckResult(
        common_columns=common_columns,
        only_in_experimental=only_in_experimental,
        only_in_simulation=only_in_simulation,
        dtype_mismatches=mismatches,
    )


def parse_column_mapping_text(mapping_text: str) -> Dict[str, str]:
    stripped = mapping_text.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Enter a JSON or Python dictionary like "
                '{"synthetic_column": "experimental_column"}.'
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Column mapping must be a dictionary.")

    return {str(source): str(target) for source, target in parsed.items()}


def normalize_mapping_editor_rows(mapping_rows: pd.DataFrame) -> Dict[str, str]:
    if mapping_rows.empty:
        return {}

    mapping = {}
    for _, row in mapping_rows.iterrows():
        source = row.get("synthetic_column")
        target = row.get("experimental_column")
        if pd.isna(source) or pd.isna(target):
            continue
        source = str(source).strip()
        target = str(target).strip()
        if source and target:
            mapping[source] = target
    return mapping


def validate_column_mapping(
    mapping: Dict[str, str],
    df_exp: pd.DataFrame,
    df_sim: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows = []
    usable_mapping = {}
    exp_cols = set(df_exp.columns)
    sim_cols = set(df_sim.columns)
    duplicate_sources = {source for source in mapping if list(mapping.keys()).count(source) > 1}
    duplicate_targets = {
        target for target in mapping.values() if list(mapping.values()).count(target) > 1
    }

    for sim_col, exp_col in mapping.items():
        sim_col = str(sim_col).strip()
        exp_col = str(exp_col).strip()
        if not sim_col or not exp_col:
            status = "Missing source or target column"
        elif sim_col in duplicate_sources:
            status = "Duplicate synthetic source in mapping"
        elif exp_col in duplicate_targets:
            status = "Duplicate experimental target in mapping"
        elif sim_col not in sim_cols:
            status = "Synthetic column missing"
        elif exp_col not in exp_cols:
            status = "Experimental target column missing"
        elif sim_col == exp_col:
            status = "Already matching; no rename needed"
        elif exp_col in sim_cols:
            status = "Target name already exists in synthetic"
        else:
            status = "Ready to apply"
            usable_mapping[sim_col] = exp_col

        rows.append(
            {
                "synthetic_column": sim_col,
                "experimental_column": exp_col,
                "status": status,
            }
        )

    return pd.DataFrame(rows), usable_mapping


def run_imbalance_analysis(
    df: pd.DataFrame,
    column: str,
    lower_threshold: float,
    upper_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if lower_threshold > upper_threshold:
        raise ValueError("Lower threshold must be less than or equal to upper threshold.")

    values = pd.to_numeric(df[column], errors="coerce")
    valid_values = values.notna()
    within_thresholds = valid_values & values.between(
        lower_threshold,
        upper_threshold,
        inclusive="both",
    )
    below_threshold = valid_values & (values < lower_threshold)
    above_threshold = valid_values & (values > upper_threshold)
    outside_thresholds = below_threshold | above_threshold

    valid_count = int(valid_values.sum())
    if valid_count == 0:
        raise ValueError("Selected column does not contain numeric values to analyze.")

    within_count = int(within_thresholds.sum())
    outside_count = int(outside_thresholds.sum())
    missing_or_non_numeric_count = int((~valid_values).sum())

    denominator = valid_count
    result = pd.DataFrame(
        [
            {
                "threshold_status": "Within thresholds",
                "count": within_count,
                "percentage": round(within_count / denominator * 100, 2),
            },
            {
                "threshold_status": "Outside thresholds",
                "count": outside_count,
                "percentage": round(outside_count / denominator * 100, 2),
            },
        ]
    )
    summary = {
        "column": column,
        "lower_threshold": float(lower_threshold),
        "upper_threshold": float(upper_threshold),
        "total_rows": int(df.shape[0]),
        "valid_numeric_parts": valid_count,
        "within_thresholds": within_count,
        "outside_thresholds": outside_count,
        "below_lower_threshold": int(below_threshold.sum()),
        "above_upper_threshold": int(above_threshold.sum()),
        "missing_or_non_numeric": missing_or_non_numeric_count,
        "within_percentage": round(within_count / denominator * 100, 2),
        "outside_percentage": round(outside_count / denominator * 100, 2),
    }
    return result, summary


def format_column_list(columns: List[str], limit: int = 10) -> str:
    if not columns:
        return "None"
    visible_columns = [str(col) for col in columns[:limit]]
    if len(columns) > limit:
        visible_columns.append(f"... +{len(columns) - limit} more")
    return ", ".join(visible_columns)


def prepare_vertical_merge_dataframes(
    df_exp: pd.DataFrame, df_sim: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    exp_columns = df_exp.columns.tolist()
    simulation_rename_map = {}
    df_sim_mapped = df_sim.copy()
    mapped_sim_columns = df_sim_mapped.columns.tolist()
    mapped_sim_column_set = set(mapped_sim_columns)

    exp_duplicate_columns = df_exp.columns[df_exp.columns.duplicated()].unique().tolist()
    sim_duplicate_columns = (
        df_sim_mapped.columns[df_sim_mapped.columns.duplicated()].unique().tolist()
    )
    common_columns = [col for col in exp_columns if col in mapped_sim_column_set]
    common_column_set = set(common_columns)
    dropped_experimental_columns = [col for col in exp_columns if col not in mapped_sim_column_set]
    dropped_simulation_columns = [col for col in mapped_sim_columns if col not in common_column_set]
    no_duplicate_columns = not exp_duplicate_columns and not sim_duplicate_columns
    can_merge = bool(common_columns) and no_duplicate_columns

    if can_merge:
        prepared_exp = df_exp[common_columns].copy()
        prepared_sim = df_sim_mapped[common_columns].copy()
    else:
        prepared_exp = pd.DataFrame()
        prepared_sim = pd.DataFrame()

    details = {
        "can_merge": can_merge,
        "simulation_rename_map": simulation_rename_map,
        "common_columns": common_columns,
        "dropped_experimental_columns": dropped_experimental_columns,
        "dropped_simulation_columns": dropped_simulation_columns,
        "experimental_duplicate_columns": exp_duplicate_columns,
        "simulation_duplicate_columns": sim_duplicate_columns,
        "prepared_experimental_shape": list(prepared_exp.shape),
        "prepared_simulation_shape": list(prepared_sim.shape),
    }
    return prepared_exp, prepared_sim, details


def check_vertical_merge_schema(
    df_exp: pd.DataFrame, df_sim: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    _, _, details = prepare_vertical_merge_dataframes(df_exp, df_sim)
    common_columns = details["common_columns"]
    dropped_experimental_columns = details["dropped_experimental_columns"]
    dropped_simulation_columns = details["dropped_simulation_columns"]
    exp_duplicate_columns = details["experimental_duplicate_columns"]
    sim_duplicate_columns = details["simulation_duplicate_columns"]
    no_duplicate_columns = not exp_duplicate_columns and not sim_duplicate_columns

    check_rows = [
        {
            "check": "Common columns kept",
            "experimental": f"{len(common_columns)} columns",
            "simulation": f"{len(common_columns)} columns",
            "status": "Pass" if common_columns else "Fail",
        },
        {
            "check": "Experimental-only columns dropped",
            "experimental": f"{len(dropped_experimental_columns)} columns",
            "simulation": "Not applicable",
            "status": "Check Step 2 analysis" if dropped_experimental_columns else "Pass",
        },
        {
            "check": "Synthetic-only columns dropped",
            "experimental": "Not applicable",
            "simulation": f"{len(dropped_simulation_columns)} columns",
            "status": "Check Step 2 analysis" if dropped_simulation_columns else "Pass",
        },
        {
            "check": "Duplicate columns after mapping",
            "experimental": format_column_list(exp_duplicate_columns),
            "simulation": format_column_list(sim_duplicate_columns),
            "status": "Pass" if no_duplicate_columns else "Fail",
        },
        {
            "check": "Prepared column count",
            "experimental": details["prepared_experimental_shape"][1],
            "simulation": details["prepared_simulation_shape"][1],
            "status": "Pass" if details["can_merge"] else "Fail",
        },
    ]
    return pd.DataFrame(check_rows), details


def merge_dataframes(
    df_exp: pd.DataFrame, df_sim: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prepared_exp, prepared_sim, details = prepare_vertical_merge_dataframes(df_exp, df_sim)
    if not details["can_merge"]:
        raise ValueError("Datasets do not have merge-ready common columns.")
    merged = pd.concat([prepared_exp, prepared_sim], ignore_index=True)
    return merged, details


def add_relative_plot_position(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.reset_index(drop=True).copy()
    if len(plot_df) <= 1:
        plot_df["plot_position"] = 0.0
    else:
        plot_df["plot_position"] = plot_df.index / (len(plot_df) - 1) * 100
    return plot_df


def build_hybrid_analysis_plot_df(
    df_exp: pd.DataFrame,
    df_sim: pd.DataFrame,
    column: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    production_values = pd.to_numeric(df_exp[column], errors="coerce")
    simulation_values = pd.to_numeric(df_sim[column], errors="coerce")

    production_plot_df = pd.DataFrame(
        {
            "original_index": df_exp.index.astype(str),
            "value": production_values,
            "source": "Production",
        }
    ).dropna(subset=["value"])
    simulation_plot_df = pd.DataFrame(
        {
            "original_index": df_sim.index.astype(str),
            "value": simulation_values,
            "source": "Synthetic",
        }
    ).dropna(subset=["value"])

    production_rows_before_sampling = len(production_plot_df)
    simulation_rows_before_sampling = len(simulation_plot_df)

    if len(production_plot_df) > MAX_PRODUCTION_PLOT_ROWS:
        production_plot_df = production_plot_df.sample(
            n=MAX_PRODUCTION_PLOT_ROWS,
            random_state=RANDOM_STATE,
        ).sort_index()
    if len(simulation_plot_df) > MAX_SIMULATION_PLOT_ROWS:
        simulation_plot_df = simulation_plot_df.sample(
            n=MAX_SIMULATION_PLOT_ROWS,
            random_state=RANDOM_STATE,
        ).sort_index()

    production_plot_df = add_relative_plot_position(production_plot_df)
    simulation_plot_df = add_relative_plot_position(simulation_plot_df)
    plot_df = pd.concat([production_plot_df, simulation_plot_df], ignore_index=True)

    sampling_details = {
        "production_available": production_rows_before_sampling,
        "simulation_available": simulation_rows_before_sampling,
        "production_plotted": len(production_plot_df),
        "simulation_plotted": len(simulation_plot_df),
    }
    return plot_df, sampling_details


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def format_optional_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def threshold_bucket_counts(
    df: pd.DataFrame,
    column: str,
    thresholds: Dict[str, float] | None,
) -> Dict[str, int]:
    if not thresholds or column not in df.columns:
        return {
            "below_lower_threshold": 0,
            "within_thresholds": 0,
            "above_upper_threshold": 0,
            "outside_thresholds": 0,
            "valid_numeric_rows": 0,
        }

    values = pd.to_numeric(df[column], errors="coerce")
    valid_values = values.notna()
    below = valid_values & (values < float(thresholds["lower"]))
    above = valid_values & (values > float(thresholds["upper"]))
    outside = below | above
    return {
        "below_lower_threshold": int(below.sum()),
        "within_thresholds": int((valid_values & ~outside).sum()),
        "above_upper_threshold": int(above.sum()),
        "outside_thresholds": int(outside.sum()),
        "valid_numeric_rows": int(valid_values.sum()),
    }


def get_thresholds_for_column(column: str) -> Dict[str, float] | None:
    session_thresholds = st.session_state.get("imbalance_thresholds")
    if st.session_state.get("imbalance_column") == column and session_thresholds:
        return {
            "lower": float(session_thresholds["lower"]),
            "upper": float(session_thresholds["upper"]),
        }

    default_thresholds = DEFAULT_IMBALANCE_THRESHOLDS.get(column)
    if default_thresholds:
        return {
            "lower": float(default_thresholds["lower"]),
            "upper": float(default_thresholds["upper"]),
        }

    return None


def get_threshold_lookup_for_column(column: str) -> Dict[str, Dict[str, float]] | None:
    thresholds = get_thresholds_for_column(column)
    return {column: thresholds} if thresholds else None


def get_preferred_numeric_column_index(numeric_columns: List[str]) -> int:
    preferred_columns = [
        st.session_state.get("imbalance_column"),
        *DEFAULT_IMBALANCE_THRESHOLDS.keys(),
    ]
    for column in preferred_columns:
        if column in numeric_columns:
            return numeric_columns.index(column)
    return 0


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        column for column in df.columns if pd.to_numeric(df[column], errors="coerce").notna().any()
    ]


def prepare_feature_selection_frame(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    selected_columns = [target_column, *feature_columns]
    numeric_df = df[selected_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([float("inf"), float("-inf")], pd.NA)
    rows_before_dropna = len(numeric_df)
    numeric_df = numeric_df.dropna(subset=selected_columns)
    rows_after_dropna = len(numeric_df)
    if rows_after_dropna < 10:
        raise ValueError("At least 10 complete numeric rows are required for feature selection.")
    if len(numeric_df) > MAX_FEATURE_SELECTION_ROWS:
        numeric_df = numeric_df.sample(
            n=MAX_FEATURE_SELECTION_ROWS,
            random_state=RANDOM_STATE,
        )
    return numeric_df, {
        "rows_before_dropna": int(rows_before_dropna),
        "rows_after_dropna": int(rows_after_dropna),
        "rows_used": int(len(numeric_df)),
    }


def run_feature_selection_analysis(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    analysis_method: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if target_column not in df.columns:
        raise ValueError("Selected target column is missing from the real-world dataset.")
    if not feature_columns:
        raise ValueError("Select at least one feature to analyze.")

    feature_columns = [col for col in feature_columns if col != target_column]
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Selected features are missing: {format_column_list(missing_features)}")

    analysis_df, preparation_details = prepare_feature_selection_frame(
        df,
        target_column,
        feature_columns,
    )
    X = analysis_df[feature_columns]
    y = analysis_df[target_column]

    if analysis_method == "Pearson correlation":
        rows = []
        for feature in feature_columns:
            correlation = X[feature].corr(y, method="pearson")
            if pd.isna(correlation):
                correlation = 0.0
            rows.append(
                {
                    "feature": feature,
                    "score": abs(float(correlation)),
                    "metric": "absolute Pearson correlation",
                }
            )
    elif analysis_method == "Random Forest feature importance":
        model = RandomForestRegressor(
            n_estimators=FEATURE_SELECTION_RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X, y)
        rows = [
            {
                "feature": feature,
                "score": float(score),
                "metric": "impurity importance",
            }
            for feature, score in zip(feature_columns, model.feature_importances_)
        ]
    elif analysis_method == "Permutation importance":
        if len(analysis_df) < 20:
            raise ValueError("Permutation importance requires at least 20 complete numeric rows.")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
        )
        model = RandomForestRegressor(
            n_estimators=FEATURE_SELECTION_RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        importance = permutation_importance(
            model,
            X_valid,
            y_valid,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rows = [
            {
                "feature": feature,
                "score": max(0.0, float(mean_score)),
                "metric": "mean score decrease",
            }
            for feature, mean_score in zip(feature_columns, importance.importances_mean)
        ]
    else:
        raise ValueError(f"Unsupported feature selection method: {analysis_method}")

    result_df = pd.DataFrame(rows).sort_values(
        ["score", "feature"],
        ascending=[False, True],
    )
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    summary = {
        **preparation_details,
        "target_column": target_column,
        "analysis_method": analysis_method,
        "analyzed_features": len(feature_columns),
        "feature_columns": feature_columns,
    }
    return result_df, summary


def apply_selected_feature_subset(
    df_exp: pd.DataFrame,
    df_sim: pd.DataFrame,
    target_column: str,
    selected_features: List[str],
    additional_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    keep_columns = []
    for column in [target_column, *selected_features, *additional_features]:
        if column in df_exp.columns and column not in keep_columns:
            keep_columns.append(column)
    if target_column not in keep_columns:
        raise ValueError("The selected target column must be kept.")
    if len(keep_columns) < 2:
        raise ValueError("Keep at least one feature in addition to the target column.")

    rejected_experimental_columns = [col for col in df_exp.columns if col not in keep_columns]
    same_name_synthetic_columns_to_drop = [
        col for col in rejected_experimental_columns if col in df_sim.columns
    ]
    selected_exp = df_exp[keep_columns].copy()
    selected_sim = df_sim.drop(columns=same_name_synthetic_columns_to_drop).copy()
    details = {
        "target_column": target_column,
        "kept_experimental_columns": keep_columns,
        "dropped_experimental_columns": rejected_experimental_columns,
        "dropped_same_name_synthetic_columns": same_name_synthetic_columns_to_drop,
    }
    return selected_exp, selected_sim, details


def sample_threshold_aware_production_df(
    df: pd.DataFrame,
    n_rows: int,
    target_column: str,
    thresholds: Dict[str, float] | None,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    n_rows = max(1, min(int(n_rows), len(df)))
    if n_rows >= len(df):
        counts = threshold_bucket_counts(df, target_column, thresholds)
        return df.copy(), {
            "method": "all_production_rows",
            "requested_rows": n_rows,
            "sampled_rows": len(df),
            **counts,
        }

    if not thresholds or target_column not in df.columns:
        sampled_df = df.sample(n=n_rows, random_state=random_state).sort_index()
        return sampled_df, {
            "method": "random_downsample_no_thresholds",
            "requested_rows": n_rows,
            "sampled_rows": len(sampled_df),
            **threshold_bucket_counts(sampled_df, target_column, thresholds),
        }

    values = pd.to_numeric(df[target_column], errors="coerce")
    valid_values = values.notna()
    below_mask = valid_values & (values < float(thresholds["lower"]))
    above_mask = valid_values & (values > float(thresholds["upper"]))
    outside_mask = below_mask | above_mask

    outside_df = df[outside_mask]
    background_df = df[~outside_mask]

    if outside_df.empty:
        sampled_df = df.sample(n=n_rows, random_state=random_state).sort_index()
    elif len(outside_df) <= n_rows:
        remaining_rows = n_rows - len(outside_df)
        if remaining_rows > 0 and not background_df.empty:
            background_sample = background_df.sample(
                n=min(remaining_rows, len(background_df)),
                random_state=random_state,
            )
            sampled_df = pd.concat([outside_df, background_sample], axis=0).sort_index()
        else:
            sampled_df = outside_df.sort_index()
    else:
        below_df = df[below_mask]
        above_df = df[above_mask]
        sampled_parts = []
        available_outside = len(outside_df)
        for bucket_df in [below_df, above_df]:
            if bucket_df.empty:
                continue
            bucket_rows = int(round(n_rows * len(bucket_df) / available_outside))
            bucket_rows = max(1, min(bucket_rows, len(bucket_df)))
            sampled_parts.append(bucket_df.sample(n=bucket_rows, random_state=random_state))
        sampled_df = pd.concat(sampled_parts, axis=0)
        if len(sampled_df) > n_rows:
            sampled_df = sampled_df.sample(n=n_rows, random_state=random_state)
        elif len(sampled_df) < n_rows:
            remaining_outside = outside_df.drop(index=sampled_df.index, errors="ignore")
            if not remaining_outside.empty:
                sampled_df = pd.concat(
                    [
                        sampled_df,
                        remaining_outside.sample(
                            n=min(n_rows - len(sampled_df), len(remaining_outside)),
                            random_state=random_state,
                        ),
                    ],
                    axis=0,
                )
        sampled_df = sampled_df.sort_index()

    counts = threshold_bucket_counts(sampled_df, target_column, thresholds)
    available_counts = threshold_bucket_counts(df, target_column, thresholds)
    return sampled_df, {
        "method": "threshold_aware_downsample",
        "requested_rows": n_rows,
        "sampled_rows": len(sampled_df),
        "available_outside_thresholds": available_counts["outside_thresholds"],
        **counts,
    }


def merge_shared_test_predictions(
    hybrid_predictions: pd.DataFrame,
    production_predictions: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = ["original_index", "source", "y_true"]
    if (
        "threshold_bucket" in hybrid_predictions.columns
        and "threshold_bucket" in production_predictions.columns
    ):
        identity_columns.append("threshold_bucket")

    hybrid_identity = hybrid_predictions[identity_columns].reset_index(drop=True)
    production_identity = production_predictions[identity_columns].reset_index(drop=True)
    if not hybrid_identity.equals(production_identity):
        raise ValueError(
            "The hybrid and real-world-only model runs did not use the same test rows. "
            "Please retry training or check the split configuration."
        )

    merged_predictions = hybrid_predictions.copy()
    production_prediction_columns = [
        col for col in production_predictions.columns if col.endswith(" prediction")
    ]
    for column in production_prediction_columns:
        merged_predictions[column] = production_predictions[column].to_numpy()
    return merged_predictions


def with_training_dataset_in_split_summary(
    split_summary: List[Dict[str, object]],
    training_dataset: str,
) -> List[Dict[str, object]]:
    return [
        {
            "training_dataset": training_dataset,
            **row,
        }
        for row in split_summary
    ]


def format_real_world_display_label(value):
    if not isinstance(value, str):
        return value
    return (
        value.replace("Production only", "Real-world only")
        .replace("Production-Only", "Real-world-Only")
        .replace("production-only", "real-world-only")
        .replace("Production", "Real-world")
        .replace("production", "real-world")
    )


def make_real_world_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df.columns = [
        str(column)
        .replace("production_", "real_world_")
        .replace("Production", "Real-world")
        .replace("production", "real-world")
        for column in display_df.columns
    ]
    for column in display_df.select_dtypes(include=["object", "string"]).columns:
        display_df[column] = display_df[column].map(format_real_world_display_label)
    return display_df


def rename_percentage_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df.columns = [
        f"{column[:-4]} (%)" if str(column).endswith("_pct") else column
        for column in display_df.columns
    ]
    return display_df


def make_prediction_preview_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.drop(columns=["source"], errors="ignore").copy()
    display_df = make_real_world_display_dataframe(display_df)
    for column in display_df.columns:
        if column == "original_index":
            continue
        numeric_values = pd.to_numeric(display_df[column], errors="coerce")
        if numeric_values.notna().any():
            display_df[column] = numeric_values.map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    return display_df


def build_model_comparison_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if "training_dataset" not in metrics_df.columns:
        return pd.DataFrame()

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    hybrid_metrics = test_metrics[test_metrics["training_dataset"] == "Hybrid"]
    production_metrics = test_metrics[
        test_metrics["training_dataset"].isin(["Production only", "Real-world only"])
    ]
    common_models = sorted(
        set(hybrid_metrics["model"]).intersection(set(production_metrics["model"]))
    )
    rows = []
    for model_name in common_models:
        hybrid_row = hybrid_metrics[hybrid_metrics["model"] == model_name].iloc[0]
        production_row = production_metrics[production_metrics["model"] == model_name].iloc[0]
        rows.append(
            {
                "model": model_name,
                "real_world_rmse": production_row["rmse"],
                "hybrid_rmse": hybrid_row["rmse"],
                "rmse_improvement": production_row["rmse"] - hybrid_row["rmse"],
                "real_world_mae": production_row["mae"],
                "hybrid_mae": hybrid_row["mae"],
                "mae_improvement": production_row["mae"] - hybrid_row["mae"],
                "real_world_r2": production_row["r2"],
                "hybrid_r2": hybrid_row["r2"],
                "r2_improvement": hybrid_row["r2"] - production_row["r2"],
            }
        )
    return pd.DataFrame(rows)


def sample_prediction_plot_df(plot_df: pd.DataFrame) -> pd.DataFrame:
    if len(plot_df) <= MAX_PREDICTION_PLOT_ROWS:
        return plot_df

    if "prediction_status" not in plot_df.columns:
        return plot_df.sample(n=MAX_PREDICTION_PLOT_ROWS, random_state=MODEL_RANDOM_STATE)

    priority_statuses = ["True Positive", "False Positive", "False Negative"]
    priority_df = plot_df[plot_df["prediction_status"].isin(priority_statuses)]
    background_df = plot_df[~plot_df["prediction_status"].isin(priority_statuses)]

    if len(priority_df) > MAX_PRIORITY_PREDICTION_PLOT_ROWS:
        priority_df = priority_df.groupby("prediction_status", group_keys=False).sample(
            frac=MAX_PRIORITY_PREDICTION_PLOT_ROWS / len(priority_df),
            random_state=MODEL_RANDOM_STATE,
        )
        background_rows = 0
    else:
        background_rows = min(
            MAX_TRUE_NEGATIVE_PREDICTION_PLOT_ROWS,
            max(0, MAX_PREDICTION_PLOT_ROWS - len(priority_df)),
        )

    if background_rows == 0:
        background_df = background_df.iloc[0:0]
    elif len(background_df) > background_rows:
        background_df = background_df.sample(n=background_rows, random_state=MODEL_RANDOM_STATE)

    return pd.concat([priority_df, background_df], ignore_index=True)


def build_prediction_plot_layers(
    predictions_df: pd.DataFrame,
    selected_prediction: str,
    thresholds: Dict[str, float] | None,
) -> alt.LayerChart:
    full_plot_df = predictions_df[["y_true", selected_prediction]].copy()
    full_plot_df["y_true"] = pd.to_numeric(full_plot_df["y_true"], errors="coerce")
    full_plot_df[selected_prediction] = pd.to_numeric(
        full_plot_df[selected_prediction],
        errors="coerce",
    )
    full_plot_df = full_plot_df.dropna().copy()

    values_for_domain = [
        float(full_plot_df["y_true"].min()),
        float(full_plot_df["y_true"].max()),
        float(full_plot_df[selected_prediction].min()),
        float(full_plot_df[selected_prediction].max()),
    ]

    lower = upper = None
    if thresholds:
        lower = float(thresholds["lower"])
        upper = float(thresholds["upper"])
        true_outside = (full_plot_df["y_true"] < lower) | (full_plot_df["y_true"] > upper)
        pred_outside = (full_plot_df[selected_prediction] < lower) | (
            full_plot_df[selected_prediction] > upper
        )
        true_inside = ~true_outside
        pred_inside = ~pred_outside
        full_plot_df["prediction_status"] = "Unclassified"
        full_plot_df.loc[true_inside & pred_inside, "prediction_status"] = "True Negative"
        full_plot_df.loc[true_outside & pred_outside, "prediction_status"] = "True Positive"
        full_plot_df.loc[true_inside & pred_outside, "prediction_status"] = "False Positive"
        full_plot_df.loc[true_outside & pred_inside, "prediction_status"] = "False Negative"
        values_for_domain.extend([lower, upper])
    else:
        full_plot_df["prediction_status"] = "Prediction"

    axis_min = min(values_for_domain)
    axis_max = max(values_for_domain)
    padding = (axis_max - axis_min) * 0.08 if axis_max > axis_min else 1.0
    axis_min -= padding
    axis_max += padding
    plot_df = sample_prediction_plot_df(full_plot_df)

    perfect_line_df = pd.DataFrame({"value": [axis_min, axis_max]})
    layers = []
    if thresholds:
        true_region_df = pd.DataFrame([{"x": lower, "x2": upper, "y": axis_min, "y2": axis_max}])
        pred_region_df = pd.DataFrame([{"x": axis_min, "x2": axis_max, "y": lower, "y2": upper}])
        layers.extend(
            [
                alt.Chart(true_region_df)
                .mark_rect(color="#2a6fbb", opacity=0.08)
                .encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q"),
                alt.Chart(pred_region_df)
                .mark_rect(color="#d95f02", opacity=0.08)
                .encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q"),
            ]
        )

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=55, opacity=0.72)
        .encode(
            x=alt.X(
                "y_true:Q",
                title="True target value",
                scale=alt.Scale(domain=[axis_min, axis_max], zero=False),
            ),
            y=alt.Y(
                f"{selected_prediction}:Q",
                title="Predicted target value",
                scale=alt.Scale(domain=[axis_min, axis_max], zero=False),
            ),
            color=alt.Color(
                "prediction_status:N",
                title="Prediction class",
                scale=alt.Scale(
                    domain=[
                        "True Negative",
                        "True Positive",
                        "False Positive",
                        "False Negative",
                        "Prediction",
                    ],
                    range=["#4d4d4d", "#2e7d32", "#f39c12", "#c62828", "#2a6fbb"],
                ),
            ),
            tooltip=[
                alt.Tooltip("y_true:Q", title="True"),
                alt.Tooltip(f"{selected_prediction}:Q", title="Predicted"),
                alt.Tooltip("prediction_status:N", title="Prediction class"),
            ],
        )
    )
    perfect_line = (
        alt.Chart(perfect_line_df)
        .mark_line(strokeDash=[6, 4], color="#c62828", size=2)
        .encode(x="value:Q", y="value:Q")
    )
    return alt.layer(*layers, scatter, perfect_line).properties(height=460)


def render_upload_step() -> None:
    st.markdown(
        "Web application for the implementation of the hybrid model generic module by integrating real-world "
        "sensor data with synthetic data to overcome the fundamental challenge of data scarcity and "
        "imbalance related to anomalies and rare defects."
    )
    st.markdown(
        "**Real-world data:** production, experimental and process datasets.  \n"
        "**Synthetic data:** high fidelity simulation results."
    )

    st.subheader(":blue[Step 1: Upload Datasets]", divider="blue")
    st.markdown(
        "Upload one real-world dataset and one synthetic dataset. "
        f"Each file must be <= {MAX_FILE_SIZE_MB} MB."
    )

    allowed = ", ".join(ALLOWED_EXTENSIONS)
    st.markdown(f"Supported file types: `{allowed}`")

    exp_file = st.file_uploader(
        "Real-world dataset",
        type=[ext[1:] for ext in ALLOWED_EXTENSIONS],
        key="exp_upload",
    )
    sim_file = st.file_uploader(
        "Synthetic dataset",
        type=[ext[1:] for ext in ALLOWED_EXTENSIONS],
        key="sim_upload",
    )

    if st.button("Load both datasets", type="primary"):
        if exp_file is None or sim_file is None:
            st.error("Please upload both files before loading.")
            return

        valid_exp, message_exp = validate_uploaded_file(exp_file)
        valid_sim, message_sim = validate_uploaded_file(sim_file)
        errors = [
            m
            for m in [message_exp if not valid_exp else "", message_sim if not valid_sim else ""]
            if m
        ]
        if errors:
            for msg in errors:
                st.error(msg)
            return

        try:
            df_exp = load_dataframe(exp_file)
            df_sim = load_dataframe(sim_file)
            st.session_state["experimental_df"] = df_exp
            st.session_state["simulation_df"] = df_sim
            reset_downstream_state()
            st.session_state["auto_run_schema_check"] = False
            st.success("Datasets loaded successfully. Moving to matching check.")
            go_to_step(2)
        except USER_FACING_ERRORS as exc:
            st.error(f"Failed to load files: {exc}")
            return

    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]
    if df_exp is not None and df_sim is not None:
        render_dataset_preview(df_exp, df_sim)


def render_schema_step() -> None:
    st.subheader(":blue[Step 2: Datasets' preprocessing]", divider="blue")
    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]

    if df_exp is None or df_sim is None:
        st.info("Please upload both datasets in Step 1.")
        return

    render_dataset_preview(df_exp, df_sim)

    st.markdown("""---""")
    st.markdown(":blue[**Step 2.1: Features' drop (optional)**]")
    st.write(
        "If there are features that should not be used in further steps, "
        "drop them here before running the matching check."
    )
    feature_drop_success_message = st.session_state.pop(
        "feature_drop_success_message",
        None,
    )
    if feature_drop_success_message:
        st.success(feature_drop_success_message)

    if st.session_state.pop("reset_feature_drop_checkbox", False):
        st.session_state["drop_columns_before_matching"] = False

    drop_columns = st.checkbox(
        "Drop features",
        key="drop_columns_before_matching",
    )
    if drop_columns:
        drop_exp_col, drop_sim_col = st.columns(2)
        with drop_exp_col:
            experimental_columns_to_drop = st.multiselect(
                "Experimental features to drop",
                df_exp.columns.tolist(),
                key=f"experimental_columns_to_drop_{len(df_exp.columns)}",
            )
        with drop_sim_col:
            simulation_columns_to_drop = st.multiselect(
                "Synthetic features to drop",
                df_sim.columns.tolist(),
                key=f"simulation_columns_to_drop_{len(df_sim.columns)}",
            )

        selected_drop_count = len(experimental_columns_to_drop) + len(simulation_columns_to_drop)
        st.write(
            f"Selected features to drop: **{selected_drop_count}** "
            f"({len(experimental_columns_to_drop)} experimental, "
            f"{len(simulation_columns_to_drop)} synthetic)."
        )
        if st.button("Apply feature drop", type="primary"):
            if selected_drop_count == 0:
                st.warning("Select at least one feature to drop before applying.")
            elif len(experimental_columns_to_drop) >= len(df_exp.columns):
                st.error("Cannot drop all features from the experimental dataset.")
            elif len(simulation_columns_to_drop) >= len(df_sim.columns):
                st.error("Cannot drop all features from the synthetic dataset.")
            else:
                if experimental_columns_to_drop:
                    df_exp = df_exp.drop(columns=experimental_columns_to_drop)
                    st.session_state["experimental_df"] = df_exp
                if simulation_columns_to_drop:
                    df_sim = df_sim.drop(columns=simulation_columns_to_drop)
                    st.session_state["simulation_df"] = df_sim
                st.session_state["schema_check"] = None
                st.session_state["auto_run_schema_check"] = False
                reset_post_schema_state()
                st.session_state["reset_feature_drop_checkbox"] = True
                st.session_state["feature_drop_success_message"] = (
                    f"Selected features were dropped successfully: "
                    f"{len(experimental_columns_to_drop)} experimental and "
                    f"{len(simulation_columns_to_drop)} synthetic. Review the updated "
                    f"preview above."
                )
                st.rerun()

    st.markdown("""---""")
    st.markdown(":blue[**Step 2.2: Feature selection analysis (optional)**]")
    st.markdown(
        "Rank numeric real-world features against a selected regression target.\n\n"
        "Recommendations:\n"
        "- **Permutation importance**: best default for predictive value.\n"
        "- **Pearson correlation**: fastest option for linear screening.\n"
        "- **Random Forest importance**: faster nonlinear ranking."
    )
    if st.session_state.pop("reset_feature_selection_checkbox", False):
        st.session_state["run_feature_selection_analysis"] = False

    feature_selection_success_message = st.session_state.pop(
        "feature_selection_success_message",
        None,
    )
    if feature_selection_success_message:
        st.success(feature_selection_success_message)

    run_feature_selection = st.checkbox(
        "Run feature selection analysis",
        key="run_feature_selection_analysis",
    )
    if run_feature_selection:
        numeric_columns = get_numeric_feature_columns(df_exp)
        if len(numeric_columns) < 2:
            st.warning(
                "At least two numeric real-world columns are required for feature selection."
            )
        else:
            target_index = get_preferred_numeric_column_index(numeric_columns)
            feature_target = st.selectbox(
                "Target column for feature selection",
                numeric_columns,
                index=int(target_index),
                key="feature_selection_target",
            )
            candidate_features = [col for col in numeric_columns if col != feature_target]
            selected_candidate_features = st.multiselect(
                "Numeric candidate features to analyze",
                candidate_features,
                default=candidate_features,
                key=f"feature_selection_candidates_{feature_target}",
            )
            analysis_method = st.radio(
                "Feature selection analysis",
                [
                    "Permutation importance",
                    "Random Forest feature importance",
                    "Pearson correlation",
                ],
                horizontal=True,
                key="feature_selection_method",
            )
            if analysis_method == "Permutation importance":
                st.info(
                    "Best default: trains a Random Forest and "
                    "measures how much validation performance drops when each feature is "
                    "shuffled. It is slower, so the analysis samples large datasets."
                )

            if st.button("Run feature selection analysis", type="secondary"):
                try:
                    result_df, summary = run_feature_selection_analysis(
                        df_exp,
                        feature_target,
                        selected_candidate_features,
                        analysis_method,
                    )
                    st.session_state["feature_selection_result_df"] = result_df
                    st.session_state["feature_selection_summary"] = summary
                    st.success("Feature selection analysis completed.")
                except USER_FACING_ERRORS as exc:
                    st.session_state["feature_selection_result_df"] = None
                    st.session_state["feature_selection_summary"] = None
                    st.error(f"Feature selection analysis failed: {exc}")

            feature_result_df = st.session_state.get("feature_selection_result_df")
            feature_summary = st.session_state.get("feature_selection_summary")
            if feature_result_df is not None and feature_summary is not None:
                is_current_analysis = (
                    feature_summary.get("target_column") == feature_target
                    and feature_summary.get("analysis_method") == analysis_method
                )
                if not is_current_analysis:
                    st.info(
                        "The displayed ranking was created with a previous target or "
                        "analysis method. Run the analysis again to refresh it."
                    )
                else:
                    st.write(
                        f"Analyzed **{feature_summary['analyzed_features']}** features "
                        f"using **{feature_summary['analysis_method']}** on "
                        f"**{feature_summary['rows_used']:,}** complete rows."
                    )
                    chart_df = feature_result_df.head(25).copy()
                    feature_chart = (
                        alt.Chart(chart_df)
                        .mark_bar(color="#1565c0")
                        .encode(
                            x=alt.X("score:Q", title="Selection score"),
                            y=alt.Y("feature:N", sort="-x", title="Feature"),
                            tooltip=[
                                alt.Tooltip("rank:Q", title="Rank"),
                                alt.Tooltip("feature:N", title="Feature"),
                                alt.Tooltip("score:Q", title="Score", format=".5f"),
                                alt.Tooltip(
                                    "signed_value:Q",
                                    title="Signed value",
                                    format=".5f",
                                ),
                                alt.Tooltip("metric:N", title="Metric"),
                            ],
                        )
                        .properties(height=max(250, min(650, len(chart_df) * 24)))
                    )
                    st.altair_chart(feature_chart, width="stretch")
                    render_dataframe(feature_result_df, width="stretch")

                    max_default_features = min(20, len(feature_result_df))
                    default_keep_features = feature_result_df.head(max_default_features)[
                        "feature"
                    ].tolist()
                    ranked_features = feature_result_df["feature"].tolist()
                    st.warning(
                        "The preselected features below correspond to the total number of "
                        "features evaluated in the analysis, up to a maximum of 20. "
                        "Please review the ranking, plot, and domain knowledge before "
                        "deciding which features to keep."
                    )
                    features_to_keep = st.multiselect(
                        "Features to keep from the analysis",
                        ranked_features,
                        default=default_keep_features,
                        key=f"features_to_keep_{feature_summary['target_column']}",
                    )
                    non_ranked_columns = [
                        col
                        for col in df_exp.columns
                        if col not in ranked_features and col != feature_summary["target_column"]
                    ]
                    additional_features_to_keep = st.multiselect(
                        "Additional non-ranked real-world features to keep",
                        non_ranked_columns,
                        default=[],
                        help=(
                            "Use this for categorical, ID, or domain-specific columns that "
                            "were not included in numeric feature selection but should "
                            "remain available."
                        ),
                        key=f"additional_features_to_keep_{feature_summary['target_column']}",
                    )
                    if st.button("Apply selected features", type="primary"):
                        try:
                            df_exp_selected, df_sim_selected, selection_details = (
                                apply_selected_feature_subset(
                                    df_exp,
                                    df_sim,
                                    feature_summary["target_column"],
                                    features_to_keep,
                                    additional_features_to_keep,
                                )
                            )
                            df_exp = df_exp_selected
                            df_sim = df_sim_selected
                            st.session_state["experimental_df"] = df_exp_selected
                            st.session_state["simulation_df"] = df_sim_selected
                            st.session_state["schema_check"] = None
                            st.session_state["auto_run_schema_check"] = False
                            reset_post_schema_state()
                            st.session_state["feature_selection_summary"] = {
                                **feature_summary,
                                "selection_applied": selection_details,
                            }
                            st.session_state["reset_feature_selection_checkbox"] = True
                            st.session_state["feature_selection_success_message"] = (
                                "Selected features were applied. Review the dataset "
                                "preview above before continuing.  \nFinal shapes: "
                                f"experimental dataset = {df_exp_selected.shape[0]:,} rows x "
                                f"{df_exp_selected.shape[1]:,} columns; synthetic dataset = "
                                f"{df_sim_selected.shape[0]:,} rows x "
                                f"{df_sim_selected.shape[1]:,} columns."
                            )
                            st.rerun()
                        except USER_FACING_ERRORS as exc:
                            st.error(f"Applying selected features failed: {exc}")

    st.markdown("""---""")
    st.markdown(":blue[**Step 2.3: Features mapping (optional)**]")
    st.markdown(
        "**Feature names in the real-world and synthetic datasets must match for proper integration. "
        "This step is essential to keep all the desired features in the final hybrid dataset.**"
    )
    st.write(
        "Best case: both uploaded datasets already use identical feature names. "
        "If not, define your own synthetic-to-experimental column map using the "
        "mapping input methods below."
    )

    if st.session_state["column_mapping_applied"]:
        applied = st.session_state["applied_column_mapping"]
        st.success(
            f"Custom column map applied to {len(applied)} synthetic columns. Review the dataset preview above to confirm the changes."
        )
        if applied:
            render_dataframe(
                pd.DataFrame(
                    [
                        {
                            "original_simulation_column": sim_col,
                            "renamed_to": exp_col,
                        }
                        for sim_col, exp_col in applied.items()
                    ]
                ),
                width="stretch",
            )

    if st.session_state.pop("reset_column_mapping_checkbox", False):
        st.session_state["use_custom_column_mapping"] = False

    use_custom_column_mapping = st.checkbox(
        "Define custom feature mapping",
        key="use_custom_column_mapping",
    )
    if use_custom_column_mapping:
        mapping_method = st.radio(
            "Mapping input method",
            ["Column names", "Dictionary"],
            horizontal=True,
            key="column_mapping_input_method",
        )

        if mapping_method == "Dictionary":
            mapping_text = st.text_area(
                "Synthetic-to-experimental mapping dictionary",
                value=st.session_state.get("column_mapping_text", "{}"),
                help=(
                    "Use synthetic column names as keys and experimental column names as values. "
                    "JSON and Python dictionary syntax are both accepted."
                ),
                key="column_mapping_text",
            )
            st.code('{"synthetic_feature": "real_world_feature"}', language="json")
            try:
                candidate_mapping = parse_column_mapping_text(mapping_text)
                mapping_parse_error = None
            except ValueError as exc:
                candidate_mapping = {}
                mapping_parse_error = str(exc)
                st.error(mapping_parse_error)
        else:
            only_in_simulation = [col for col in df_sim.columns if col not in df_exp.columns]
            editor_row_count = min(5, len(only_in_simulation))
            editor_default = pd.DataFrame(
                {
                    "synthetic_column": pd.Series(
                        [str(col) for col in only_in_simulation[:editor_row_count]],
                        dtype="string",
                    ),
                    "experimental_column": pd.Series(
                        [""] * editor_row_count,
                        dtype="string",
                    ),
                }
            )
            edited_mapping_rows = st.data_editor(
                editor_default,
                column_config={
                    "synthetic_column": st.column_config.SelectboxColumn(
                        "Synthetic column",
                        options=["", *map(str, df_sim.columns.tolist())],
                        required=False,
                    ),
                    "experimental_column": st.column_config.SelectboxColumn(
                        "Experimental column",
                        options=["", *map(str, df_exp.columns.tolist())],
                        required=False,
                    ),
                },
                num_rows="dynamic",
                hide_index=True,
                key="column_mapping_editor_rows_v2",
                width="stretch",
            )
            candidate_mapping = normalize_mapping_editor_rows(edited_mapping_rows)
            mapping_parse_error = None

        mapping_status_df, usable_mapping = validate_column_mapping(
            candidate_mapping,
            df_exp,
            df_sim,
        )
        if candidate_mapping:
            with st.expander("Review custom feature map", expanded=True):
                render_dataframe(mapping_status_df, width="stretch")
        else:
            st.info("No custom feature mapping is currently defined.")

        if st.button(
            "Apply custom column map",
            disabled=bool(mapping_parse_error) or not usable_mapping,
            type="secondary",
        ):
            df_sim = df_sim.rename(columns=usable_mapping)
            st.session_state["simulation_df"] = df_sim
            st.session_state["applied_column_mapping"] = usable_mapping
            st.session_state["column_mapping_applied"] = True
            st.session_state["schema_check"] = None
            st.session_state["auto_run_schema_check"] = False
            reset_post_schema_state()
            st.session_state["reset_column_mapping_checkbox"] = True
            st.success(
                f"Applied {len(usable_mapping)} custom column mappings to the synthetic dataset. "
                "Run the matching check to validate the updated columns."
            )
            st.rerun()

    else:
        st.info("No custom feature mapping is currently being edited.")

    st.markdown("""---""")
    st.markdown(":blue[**Step 2.4: Matching check**]")
    st.markdown(
        "This step checks for common and unique features between the real-world and synthetic datasets, "
        "as well as any dtype mismatches in the common features. "
        "If any issues are detected, please review the previous steps to resolve them before proceeding."
    )

    if st.button("Run matching check", type="primary"):
        st.session_state["schema_check"] = check_schema(df_exp, df_sim)

    result = st.session_state["schema_check"]
    if result is None:
        return

    st.write(f"Common features: **{len(result.common_columns)}**")
    st.write(f"Features only in experimental dataset: **{len(result.only_in_experimental)}**")
    st.write(f"Features only in synthetic dataset: **{len(result.only_in_simulation)}**")
    st.write(f"Dtype mismatches in common features: **{len(result.dtype_mismatches)}**")

    if result.common_columns:
        st.success("Common features detected.")
        render_dataframe(pd.DataFrame({"feature": result.common_columns}), width="stretch")
    if result.only_in_experimental:
        st.warning("Features only in experimental dataset detected.")
        render_dataframe(pd.DataFrame({"feature": result.only_in_experimental}), width="stretch")
    if result.only_in_simulation:
        st.warning("Features only in synthetic dataset detected.")
        render_dataframe(pd.DataFrame({"feature": result.only_in_simulation}), width="stretch")
    if result.dtype_mismatches:
        st.warning("Feature format mismatch (dtype mismatch) detected.")
        render_dataframe(pd.DataFrame(result.dtype_mismatches), width="stretch")

    if result.all_match:
        st.success("All shared features and dtypes match.")
    else:
        st.info("Warnings above should be reviewed before merge/training.")

    if st.button("Next: Imbalance analysis"):
        go_to_step(3)


def render_imbalance_step() -> None:
    st.subheader(":blue[Step 3: Imbalance Analysis]", divider="blue")
    df_exp = st.session_state["experimental_df"]

    if df_exp is None:
        st.info("Please upload the experimental dataset in Step 1.")
        return

    st.write(
        "This analysis runs on the real-world dataset only. "
        "Select a target feature, set acceptable lower and upper thresholds, "
        "then count how many parts are within or outside those limits."
    )

    experimental_columns = get_numeric_model_columns(df_exp)
    if not experimental_columns:
        st.warning("No numeric columns are available for imbalance analysis.")
        return

    default_index = get_preferred_numeric_column_index(experimental_columns)
    column = st.selectbox(
        "Target feature to analyze",
        experimental_columns,
        index=int(default_index),
    )

    default_thresholds = DEFAULT_IMBALANCE_THRESHOLDS.get(column)
    numeric_values = pd.to_numeric(df_exp[column], errors="coerce").dropna()
    fallback_lower = float(numeric_values.min()) if not numeric_values.empty else 0.0
    fallback_upper = float(numeric_values.max()) if not numeric_values.empty else 1.0
    lower_default = default_thresholds["lower"] if default_thresholds else fallback_lower
    upper_default = default_thresholds["upper"] if default_thresholds else fallback_upper

    if default_thresholds:
        st.info(
            f"Using suggested thresholds for `{column}`: "
            f"lower {lower_default}, upper {upper_default}."
        )
    else:
        st.info(
            "No predefined thresholds exist for this column. "
            "The fields below start with the observed numeric min/max."
        )

    lower_col, upper_col = st.columns(2)
    with lower_col:
        lower_threshold = st.number_input(
            "Lower threshold",
            value=float(lower_default),
            step=0.1,
            format="%.4f",
            key=f"imbalance_lower_threshold_{column}",
        )
    with upper_col:
        upper_threshold = st.number_input(
            "Upper threshold",
            value=float(upper_default),
            step=0.1,
            format="%.4f",
            key=f"imbalance_upper_threshold_{column}",
        )

    if st.button("Analyze imbalance", type="primary"):
        try:
            result_df, summary = run_imbalance_analysis(
                df_exp,
                column,
                float(lower_threshold),
                float(upper_threshold),
            )
            st.session_state["imbalance_result_df"] = result_df
            st.session_state["imbalance_ratio"] = None
            st.session_state["imbalance_column"] = column
            st.session_state["imbalance_thresholds"] = {
                "lower": float(lower_threshold),
                "upper": float(upper_threshold),
            }
            st.session_state["imbalance_summary"] = summary
        except USER_FACING_ERRORS as exc:
            st.session_state["imbalance_result_df"] = None
            st.session_state["imbalance_ratio"] = None
            st.session_state["imbalance_column"] = None
            st.session_state["imbalance_thresholds"] = None
            st.session_state["imbalance_summary"] = None
            st.error(f"Threshold analysis failed: {exc}")

    result_df = st.session_state["imbalance_result_df"]
    summary = st.session_state["imbalance_summary"]
    if result_df is not None and summary is not None:
        render_dataframe(result_df, width="stretch")

        analyzed_column = str(summary["column"])
        lower_threshold = float(summary["lower_threshold"])
        upper_threshold = float(summary["upper_threshold"])
        plot_values = pd.to_numeric(df_exp[analyzed_column], errors="coerce")
        scatter_df = pd.DataFrame(
            {
                "part_index": range(len(plot_values)),
                "original_index": df_exp.index.astype(str),
                "value": plot_values,
            }
        ).dropna(subset=["value"])
        scatter_df["threshold_status"] = scatter_df["value"].between(
            lower_threshold,
            upper_threshold,
            inclusive="both",
        )
        scatter_df["threshold_status"] = scatter_df["threshold_status"].map(
            {True: "Within thresholds", False: "Outside thresholds"}
        )
        within_plot_df = scatter_df[scatter_df["threshold_status"] == "Within thresholds"]
        outside_plot_df = scatter_df[scatter_df["threshold_status"] == "Outside thresholds"]
        sampled_within_count = 0
        if len(within_plot_df) > MAX_WITHIN_THRESHOLD_PLOT_POINTS:
            sampled_within_count = len(within_plot_df) - MAX_WITHIN_THRESHOLD_PLOT_POINTS
            within_plot_df = within_plot_df.sample(
                n=MAX_WITHIN_THRESHOLD_PLOT_POINTS,
                random_state=42,
            ).sort_values("part_index")
        scatter_df = pd.concat([within_plot_df, outside_plot_df], ignore_index=True)

        if sampled_within_count > 0:
            st.info(
                "The chart shows all outside-threshold points and a deterministic "
                f"sample of {MAX_WITHIN_THRESHOLD_PLOT_POINTS:,} within-threshold "
                f"points. The table and metrics use the full dataset."
            )

        threshold_df = pd.DataFrame(
            [
                {"threshold_label": "Lower threshold", "threshold_value": lower_threshold},
                {"threshold_label": "Upper threshold", "threshold_value": upper_threshold},
            ]
        )
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=58, opacity=0.85)
            .encode(
                x=alt.X("part_index:Q", title="Row index"),
                y=alt.Y("value:Q", title=analyzed_column),
                color=alt.Color(
                    "threshold_status:N",
                    title="Status",
                    scale=alt.Scale(
                        domain=["Within thresholds", "Outside thresholds"],
                        range=["#2e7d32", "#c62828"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("part_index:Q", title="Row index"),
                    alt.Tooltip("original_index:N", title="Original index"),
                    alt.Tooltip("value:Q", title=analyzed_column),
                    alt.Tooltip("threshold_status:N", title="Status"),
                ],
            )
        )
        threshold_lines = (
            alt.Chart(threshold_df)
            .mark_rule(strokeDash=[6, 4], size=2)
            .encode(
                y=alt.Y("threshold_value:Q"),
                color=alt.Color(
                    "threshold_label:N",
                    title="Thresholds",
                    scale=alt.Scale(
                        domain=["Lower threshold", "Upper threshold"],
                        range=["#7315c0", "#1565c0"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("threshold_label:N", title="Threshold"),
                    alt.Tooltip("threshold_value:Q", title="Value"),
                ],
            )
        )
        chart = (
            alt.layer(scatter, threshold_lines)
            .resolve_scale(color="independent")
            .properties(height=420)
        )
        st.altair_chart(chart, width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid numeric parts", summary["valid_numeric_parts"])
        m2.metric("Within thresholds", summary["within_thresholds"])
        m3.metric("Outside thresholds", summary["outside_thresholds"])
        m4.metric("Missing/non-numeric", summary["missing_or_non_numeric"])

        st.write(
            f"Applied thresholds for `{summary['column']}`: "
            f"lower **{summary['lower_threshold']}**, "
            f"upper **{summary['upper_threshold']}**."
        )
        st.write(
            f"Outside thresholds breakdown: **{summary['below_lower_threshold']}** below lower threshold, "
            f"**{summary['above_upper_threshold']}** above upper threshold."
        )
        if summary["missing_or_non_numeric"] > 0:
            st.warning(
                "Some rows were missing or non-numeric and were excluded from the "
                "within/outside percentages."
            )
        if summary["outside_thresholds"] > 0:
            st.warning("Some experimental parts are outside the selected thresholds.")
        else:
            st.success("All valid experimental parts are within the selected thresholds.")

    if st.button("Next: Merge datasets", disabled=result_df is None):
        go_to_step(4)


def render_merge_step() -> None:
    st.subheader(":blue[Step 4: Create the Hybrid Dataset]", divider="blue")
    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]

    if df_exp is None or df_sim is None:
        st.info("Please upload datasets in Step 1.")
        return

    st.write(
        "This step merges the real-world dataset with the synthetic dataset. "
        "Common features between the datasets are aligned, and unique features, "
        "not common to both datasets, are dropped before merging."
    )

    c1, c2 = st.columns(2)
    c1.metric("Experimental shape", f"{df_exp.shape[0]:,} rows x {df_exp.shape[1]:,} columns")
    c2.metric("Synthetic shape", f"{df_sim.shape[0]:,} rows x {df_sim.shape[1]:,} columns")

    schema_check_df, merge_schema_details = check_vertical_merge_schema(df_exp, df_sim)
    can_merge = bool(merge_schema_details["can_merge"])
    st.markdown("**Vertical merge schema check**")
    render_dataframe(schema_check_df, width="stretch")

    if can_merge:
        prepared_exp_shape = merge_schema_details["prepared_experimental_shape"]
        prepared_sim_shape = merge_schema_details["prepared_simulation_shape"]
        st.success(
            "Schema check passed. The datasets are ready for vertical merge after "
            "dropping non-common columns."
        )
        st.write(
            f"Merge-ready experimental shape: **{prepared_exp_shape[0]:,} rows x "
            f"{prepared_exp_shape[1]:,} columns**"
        )
        st.write(
            f"Merge-ready synthetic shape: **{prepared_sim_shape[0]:,} rows x "
            f"{prepared_sim_shape[1]:,} columns**"
        )
    else:
        st.error(
            "Schema check failed. The datasets need at least one common column and "
            "must not contain duplicate column names after optional custom mapping."
        )

    if st.button("Vertically merge datasets", disabled=not can_merge, type="primary"):
        try:
            merged, merge_schema_details = merge_dataframes(df_exp, df_sim)
            st.session_state["merged_df"] = merged
            st.session_state["merge_details"] = {
                "method": "vertical_concat",
                "experimental_rows": int(df_exp.shape[0]),
                "simulation_rows": int(df_sim.shape[0]),
                "simulation_rename_map": merge_schema_details["simulation_rename_map"],
                "common_columns": merge_schema_details["common_columns"],
                "dropped_experimental_columns": merge_schema_details[
                    "dropped_experimental_columns"
                ],
                "dropped_simulation_columns": merge_schema_details["dropped_simulation_columns"],
                "rows": int(merged.shape[0]),
                "columns": int(merged.shape[1]),
            }
            st.session_state["hybrid_analysis_completed"] = False
            st.session_state["model_metrics_df"] = None
            st.session_state["model_predictions_df"] = None
            st.session_state["training_details"] = None
            st.session_state["trained_models"] = None
            st.success("Datasets merged successfully.")
        except USER_FACING_ERRORS as exc:
            st.error(f"Merge failed: {exc}")

    merged_df = st.session_state["merged_df"]
    if merged_df is not None:
        st.markdown("**Hybrid dataset overview**")
        st.write(
            f"Hybrid dataset shape: **{merged_df.shape[0]:,} rows x "
            f"{merged_df.shape[1]:,} columns**"
        )
        render_dataframe(merged_df.head(20), width="stretch")
        st.download_button(
            label="Download hybrid dataset (CSV)",
            data=dataframe_to_csv_bytes(merged_df),
            file_name="hybrid_dataset.csv",
            mime="text/csv",
        )

    if st.button("Next: Analyze hybrid dataset", disabled=merged_df is None):
        go_to_step(5)


def render_hybrid_analysis_step() -> None:
    st.subheader(":blue[Step 5: Analyze the Hybrid Dataset]", divider="blue")
    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]
    merged_df = st.session_state["merged_df"]
    merge_details = st.session_state["merge_details"]

    if df_exp is None or df_sim is None or merged_df is None or merge_details is None:
        st.info("Please create the hybrid dataset in Step 4 before analysis.")
        return

    prepared_exp, prepared_sim, details = prepare_vertical_merge_dataframes(df_exp, df_sim)
    if not details["can_merge"]:
        st.error("The current datasets are no longer merge-ready. Please return to Step 4.")
        return

    balance_details = merge_details.get("balance") if isinstance(merge_details, dict) else None
    current_production_rows = (
        int(balance_details["experimental_rows_used"])
        if balance_details
        else int(prepared_exp.shape[0])
    )
    current_simulation_rows = (
        int(balance_details["simulation_rows_used"])
        if balance_details
        else int(prepared_sim.shape[0])
    )
    current_total_rows = current_production_rows + current_simulation_rows
    current_simulation_share = (
        current_simulation_rows / current_total_rows * 100 if current_total_rows else 0
    )
    if balance_details:
        analysis_exp = prepared_exp.sample(
            n=min(current_production_rows, prepared_exp.shape[0]),
            random_state=int(balance_details["random_state"]),
        ).sort_index()
    else:
        analysis_exp = prepared_exp
    analysis_sim = prepared_sim

    c1, c2, c3 = st.columns(3)
    c1.metric("Real-world rows in the hybrid dataset", f"{current_production_rows:,}")
    c2.metric("Synthetic rows in the hybrid dataset", f"{current_simulation_rows:,}")
    c3.metric("Synthetic data share", f"{current_simulation_share:.1f}%")

    enrichment_ratio = (
        prepared_sim.shape[0] / prepared_exp.shape[0] * 100 if prepared_exp.shape[0] > 0 else 0
    )
    st.write(
        f"Synthetic data adds **{prepared_sim.shape[0]:,} rows**, equivalent to "
        f"**{enrichment_ratio:.1f}%** of the real-world dataset size."
    )
    hybrid_balance_success_message = st.session_state.pop(
        "hybrid_balance_success_message",
        None,
    )
    if hybrid_balance_success_message:
        st.success(hybrid_balance_success_message)

    elif current_simulation_share < 50:
        st.warning(
            "Synthetic data represents less than 50% of the current hybrid dataset. "
            "You can balance the hybrid dataset by randomly dropping production rows "
            "while keeping all synthetic rows."
        )
    target_simulation_share = st.slider(
        "Target synthetic share in the hybrid dataset",
        min_value=10,
        max_value=90,
        value=50,
        step=5,
        format="%d%%",
    )
    target_fraction = target_simulation_share / 100
    target_production_rows = int(prepared_sim.shape[0] * (1 - target_fraction) / target_fraction)
    target_production_rows = max(1, min(target_production_rows, prepared_exp.shape[0]))
    resulting_total_rows = target_production_rows + prepared_sim.shape[0]
    resulting_simulation_share = prepared_sim.shape[0] / resulting_total_rows * 100
    st.write(
        f"Balancing will keep **{prepared_sim.shape[0]:,}** synthetic rows and "
        f"sample **{target_production_rows:,}** production rows, resulting in "
        f"**{resulting_total_rows:,}** hybrid rows with "
        f"**{resulting_simulation_share:.1f}%** synthetic data."
    )
    if st.button("Create balanced hybrid dataset", type="primary"):
        balanced_exp = prepared_exp.sample(
            n=target_production_rows,
            random_state=RANDOM_STATE,
        ).sort_index()
        balanced_merged = pd.concat([balanced_exp, prepared_sim], ignore_index=True)
        st.session_state["merged_df"] = balanced_merged
        st.session_state["merge_details"] = {
            **merge_details,
            "rows": int(balanced_merged.shape[0]),
            "columns": int(balanced_merged.shape[1]),
            "balance": {
                "method": "downsample_experimental",
                "target_simulation_share": float(target_simulation_share),
                "actual_simulation_share": float(resulting_simulation_share),
                "experimental_rows_available": int(prepared_exp.shape[0]),
                "experimental_rows_used": int(target_production_rows),
                "simulation_rows_used": int(prepared_sim.shape[0]),
                "random_state": int(RANDOM_STATE),
            },
        }
        st.session_state["hybrid_analysis_completed"] = False
        st.session_state["model_metrics_df"] = None
        st.session_state["model_predictions_df"] = None
        st.session_state["training_details"] = None
        st.session_state["trained_models"] = None
        st.session_state["hybrid_balance_success_message"] = "Balanced hybrid dataset created."
        st.rerun()

    common_columns = details["common_columns"]
    numeric_columns = [
        col
        for col in common_columns
        if pd.to_numeric(analysis_exp[col], errors="coerce").notna().any()
        or pd.to_numeric(analysis_sim[col], errors="coerce").notna().any()
    ]
    if not numeric_columns:
        st.warning("No numeric common columns are available for scatter-plot analysis.")
        if st.button("Next: Train models"):
            st.session_state["hybrid_analysis_completed"] = True
            go_to_step(6)
        return

    default_index = get_preferred_numeric_column_index(numeric_columns)
    selected_column = st.selectbox(
        "Feature to analyze",
        numeric_columns,
        index=int(default_index),
    )
    selected_thresholds = get_thresholds_for_column(selected_column)

    plot_df, sampling_details = build_hybrid_analysis_plot_df(
        analysis_exp,
        analysis_sim,
        selected_column,
    )
    if plot_df.empty:
        st.warning("The selected column has no numeric values to plot.")
    else:
        st.write(
            f"Plot input for `{selected_column}`: "
            f"**{sampling_details['production_plotted']:,}** production points and "
            f"**{sampling_details['simulation_plotted']:,}** synthetic points shown."
        )
        if sampling_details["simulation_available"] == 0:
            st.warning(
                "No synthetic rows contain numeric values for the selected column, so "
                "synthetic data cannot appear in this scatter plot. Try another "
                "numeric common column or review Step 2 column mapping."
            )

        if (
            sampling_details["production_available"] > sampling_details["production_plotted"]
            or sampling_details["simulation_available"] > sampling_details["simulation_plotted"]
        ):
            st.info(
                "The scatter plot is sampled for responsiveness: "
                f"production {sampling_details['production_plotted']:,}/"
                f"{sampling_details['production_available']:,}, simulation "
                f"{sampling_details['simulation_plotted']:,}/"
                f"{sampling_details['simulation_available']:,}. Summary metrics use "
                "the full merge-ready datasets."
            )

        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=42)
            .encode(
                x=alt.X(
                    "plot_position:Q",
                    title="Relative plot index within each source (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                y=alt.Y("value:Q", title=selected_column),
                color=alt.Color(
                    "source:N",
                    legend=alt.Legend(title="Dataset source", orient="top"),
                    scale=alt.Scale(
                        domain=["Production", "Synthetic"],
                        range=["#2a6fbb", "#d95f02"],
                    ),
                ),
                opacity=alt.Opacity(
                    "source:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=["Production", "Synthetic"],
                        range=[0.28, 0.78],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("original_index:N", title="Original index"),
                    alt.Tooltip("plot_position:Q", title="Relative index (%)", format=".2f"),
                    alt.Tooltip("value:Q", title=selected_column),
                ],
            )
        )
        if selected_thresholds:
            threshold_df = pd.DataFrame(
                [
                    {
                        "threshold_label": "Lower threshold",
                        "threshold_value": float(selected_thresholds["lower"]),
                    },
                    {
                        "threshold_label": "Upper threshold",
                        "threshold_value": float(selected_thresholds["upper"]),
                    },
                ]
            )
            threshold_lines = (
                alt.Chart(threshold_df)
                .mark_rule(strokeDash=[6, 4], size=2)
                .encode(
                    y=alt.Y("threshold_value:Q"),
                    color=alt.Color(
                        "threshold_label:N",
                        legend=alt.Legend(title="Thresholds", orient="top"),
                        scale=alt.Scale(
                            domain=["Lower threshold", "Upper threshold"],
                            range=["#7315c0", "#1565c0"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("threshold_label:N", title="Threshold"),
                        alt.Tooltip("threshold_value:Q", title="Value"),
                    ],
                )
            )
            chart = (
                alt.layer(scatter, threshold_lines)
                .resolve_scale(color="independent")
                .properties(height=430)
            )
        else:
            chart = scatter.properties(height=430)
            st.info("No configured threshold lines are available for this column.")

        st.altair_chart(chart, width="stretch")

    production_values = pd.to_numeric(analysis_exp[selected_column], errors="coerce")
    simulation_values = pd.to_numeric(analysis_sim[selected_column], errors="coerce")
    stats_df = pd.DataFrame(
        [
            {
                "source": "Production",
                "valid_numeric_rows": int(production_values.notna().sum()),
                "min": format_optional_float(production_values.min()),
                "median": format_optional_float(production_values.median()),
                "mean": format_optional_float(production_values.mean()),
                "max": format_optional_float(production_values.max()),
            },
            {
                "source": "Synthetic",
                "valid_numeric_rows": int(simulation_values.notna().sum()),
                "min": format_optional_float(simulation_values.min()),
                "median": format_optional_float(simulation_values.median()),
                "mean": format_optional_float(simulation_values.mean()),
                "max": format_optional_float(simulation_values.max()),
            },
        ]
    )
    st.markdown("**Distribution overview**")
    render_dataframe(stats_df, width="stretch")

    if st.button("Next: Train models"):
        st.session_state["hybrid_analysis_completed"] = True
        go_to_step(6)


def render_training_step() -> None:
    st.subheader(":blue[Step 6: Train Regression Models]", divider="blue")
    merged_df = st.session_state["merged_df"]
    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]
    merge_details = st.session_state["merge_details"]
    if merged_df is None:
        st.info("Please merge datasets in Step 4 before training.")
        return
    if df_exp is None or df_sim is None or merge_details is None:
        st.info("Please upload and merge both datasets before training.")
        return

    if not st.session_state["hybrid_analysis_completed"]:
        st.info("Please complete Step 5 before training models.")
        return

    st.write(
        "This step trains regression models for the selected target.  \n"
        "Models are trained on both the hybrid dataset and the real-world-only dataset, "
        "then compares their performance on a shared real-world test split.  \n"
        "Select below the target, feature columns, regression models to train, "
        "and the train/validation/test split ratios"
    )

    try:
        prepared_exp, prepared_sim, merge_prep_details = prepare_vertical_merge_dataframes(
            df_exp,
            df_sim,
        )
    except USER_FACING_ERRORS as exc:
        st.error(f"Could not prepare real-world/synthetic datasets for training: {exc}")
        return
    if not merge_prep_details["can_merge"]:
        st.error("The current datasets are no longer merge-ready. Please return to Step 4.")
        return

    balance_details = merge_details.get("balance") if isinstance(merge_details, dict) else None
    requested_production_rows = (
        min(int(balance_details["experimental_rows_used"]), prepared_exp.shape[0])
        if balance_details
        else prepared_exp.shape[0]
    )
    simulation_training_df = prepared_sim

    try:
        numeric_columns = get_numeric_model_columns(merged_df)
    except USER_FACING_ERRORS as exc:
        st.error(f"Could not prepare regression columns: {exc}")
        return

    if len(numeric_columns) < 2:
        st.error("At least two numeric columns are required for regression training.")
        return

    target_index = get_preferred_numeric_column_index(numeric_columns)
    target_column = st.selectbox(
        "Target column",
        numeric_columns,
        index=int(target_index),
    )
    training_thresholds = get_thresholds_for_column(target_column)
    training_threshold_lookup = get_threshold_lookup_for_column(target_column)
    if balance_details:
        production_training_df, production_sampling_details = sample_threshold_aware_production_df(
            prepared_exp,
            requested_production_rows,
            target_column,
            training_thresholds,
            random_state=int(balance_details["random_state"]),
        )
    else:
        production_training_df = prepared_exp
        production_sampling_details = {
            "method": "all_production_rows",
            "requested_rows": prepared_exp.shape[0],
            "sampled_rows": prepared_exp.shape[0],
            **threshold_bucket_counts(prepared_exp, target_column, training_thresholds),
        }

    available_feature_columns = [col for col in numeric_columns if col != target_column]
    feature_columns = st.multiselect(
        "Feature columns",
        available_feature_columns,
        default=available_feature_columns,
        key=f"regression_features_{target_column}",
    )
    ann_available = is_ann_available()
    regression_model_options = [
        model_name
        for model_name in AVAILABLE_REGRESSION_MODELS
        if model_name != ANN_MODEL_NAME or ann_available
    ]
    selected_model_names = st.multiselect(
        "Regression models to train",
        regression_model_options,
        default=regression_model_options,
    )

    st.markdown("Train / validation / test split")
    r1, r2, r3 = st.columns(3)
    with r1:
        train_ratio = st.number_input(
            "Train ratio",
            min_value=0.05,
            max_value=0.90,
            value=float(TRAIN_RATIO),
            step=0.05,
            format="%.2f",
        )
    with r2:
        val_ratio = st.number_input(
            "Validation ratio",
            min_value=0.05,
            max_value=0.90,
            value=float(VAL_RATIO),
            step=0.05,
            format="%.2f",
        )
    with r3:
        test_ratio = st.number_input(
            "Test ratio",
            min_value=0.05,
            max_value=0.90,
            value=float(TEST_RATIO),
            step=0.05,
            format="%.2f",
        )
    ratio_sum = train_ratio + val_ratio + test_ratio
    normalized_train_ratio = train_ratio / ratio_sum
    normalized_val_ratio = val_ratio / ratio_sum
    normalized_test_ratio = test_ratio / ratio_sum
    st.write(
        "Ratios are normalized before splitting. **Effective split: "
        f"train {normalized_train_ratio:.2f}, validation {normalized_val_ratio:.2f}, "
        f"test {normalized_test_ratio:.2f}.**"
    )

    if training_thresholds and production_sampling_details["outside_thresholds"] == 0:
        st.warning(
            "No real-world rows outside the selected target thresholds are available in "
            "the real-world subset used for training/testing. The test split cannot "
            "contain outside-threshold real-world points unless they exist before the split."
        )
    if not training_thresholds:
        st.info(
            "No thresholds are configured for the selected target. Training will use "
            "random splits and prediction plots will not classify threshold outcomes."
        )

    with st.expander("Training setup", expanded=True):
        setup_df = pd.DataFrame(
            [
                {"setting": "Selected target", "value": target_column},
                {"setting": "Selected features", "value": len(feature_columns)},
                {"setting": "Selected models", "value": ", ".join(selected_model_names)},
                {"setting": "Shared test source", "value": "Real-world only"},
                {
                    "setting": "Real-world rows available",
                    "value": f"{production_training_df.shape[0]:,}",
                },
                {
                    "setting": "Synthetic rows available",
                    "value": f"{simulation_training_df.shape[0]:,}",
                },
                {"setting": "Train ratio", "value": round(normalized_train_ratio, 4)},
                {"setting": "Validation ratio", "value": round(normalized_val_ratio, 4)},
                {"setting": "Test ratio", "value": round(normalized_test_ratio, 4)},
            ]
        )
        render_dataframe(setup_df, width="stretch")

    if ann_available:
        try:
            import torch

            st.caption(f"PyTorch available: `{torch.__version__}`")
        except ImportError:
            st.warning(
                "ANN is optional, but PyTorch could not be imported. "
                f"{ANN_INSTALL_GUIDANCE}"
            )
    else:
        st.info(
            "ANN is optional and is not enabled in this environment. "
            f"{ANN_INSTALL_GUIDANCE}"
        )

    if st.button(
        "Train and compare regression models",
        disabled=not feature_columns or not selected_model_names,
        type="primary",
    ):
        try:
            with st.spinner(
                f"Training hybrid and real-world-only {', '.join(selected_model_names)}..."
            ):
                hybrid_result = train_and_evaluate_regression_models(
                    merged_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    selected_model_names=selected_model_names,
                    production_df=production_training_df,
                    simulation_df=simulation_training_df,
                    thresholds=training_threshold_lookup,
                    training_dataset_label="Hybrid",
                    test_source_label="Real-world",
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    test_ratio=float(test_ratio),
                    random_state=MODEL_RANDOM_STATE,
                )
                production_result = train_and_evaluate_regression_models(
                    production_training_df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    selected_model_names=selected_model_names,
                    production_df=production_training_df,
                    simulation_df=simulation_training_df.iloc[0:0].copy(),
                    thresholds=training_threshold_lookup,
                    training_dataset_label="Real-world only",
                    test_source_label="Real-world",
                    train_ratio=float(train_ratio),
                    val_ratio=float(val_ratio),
                    test_ratio=float(test_ratio),
                    random_state=MODEL_RANDOM_STATE,
                )
                predictions_df = merge_shared_test_predictions(
                    hybrid_result.predictions_df,
                    production_result.predictions_df,
                )
                metrics_df = pd.concat(
                    [hybrid_result.metrics_df, production_result.metrics_df],
                    ignore_index=True,
                )
                split_warnings = list(
                    dict.fromkeys(
                        (hybrid_result.details.get("split_warnings") or [])
                        + (production_result.details.get("split_warnings") or [])
                    )
                )
                model_training_warnings = (
                    hybrid_result.details.get("model_training_warnings") or []
                ) + (production_result.details.get("model_training_warnings") or [])
                threshold_split_summary = with_training_dataset_in_split_summary(
                    hybrid_result.details.get("threshold_split_summary") or [],
                    "Hybrid",
                ) + with_training_dataset_in_split_summary(
                    production_result.details.get("threshold_split_summary") or [],
                    "Real-world only",
                )
                training_details = {
                    **hybrid_result.details,
                    "comparison_mode": "hybrid_vs_production_only",
                    "shared_test_rows": int(predictions_df.shape[0]),
                    "production_sampling_details": production_sampling_details,
                    "hybrid_training_details": hybrid_result.details,
                    "production_only_training_details": production_result.details,
                    "split_warnings": split_warnings,
                    "model_training_warnings": model_training_warnings,
                    "threshold_split_summary": threshold_split_summary,
                }
                trained_models = {
                    **hybrid_result.artifacts,
                    **production_result.artifacts,
                }
            st.session_state["model_metrics_df"] = metrics_df
            st.session_state["model_predictions_df"] = predictions_df
            st.session_state["training_details"] = training_details
            st.session_state["trained_models"] = trained_models
            st.success("Regression training completed.")
        except USER_FACING_ERRORS as exc:
            st.error(f"Training failed: {exc}")

    metrics_df = st.session_state["model_metrics_df"]
    predictions_df = st.session_state["model_predictions_df"]
    if metrics_df is not None:
        training_details = st.session_state["training_details"] or {}
        split_warnings = training_details.get("split_warnings") or []
        for split_warning in split_warnings:
            st.warning(format_real_world_display_label(split_warning))
        model_training_warnings = training_details.get("model_training_warnings") or []
        for model_warning in model_training_warnings:
            st.warning(
                f"{format_real_world_display_label(model_warning['training_dataset'])} - {model_warning['model']}: "
                f"{model_warning['warning']}"
            )

        split_summary = training_details.get("threshold_split_summary") or []
        if split_summary:
            with st.expander("Threshold split balance"):
                split_summary_df = pd.DataFrame(split_summary)
                if "percentage" in split_summary_df.columns:
                    split_summary_df["percentage"] = split_summary_df["percentage"].round(2)
                render_dataframe(
                    make_real_world_display_dataframe(split_summary_df), width="stretch"
                )

        comparison_df = build_model_comparison_df(metrics_df)
        if not comparison_df.empty:
            st.markdown("""---""")
            st.markdown(":blue[**Prediction results:**]")
            st.markdown("**Hybrid vs Real-world-Only Test Comparison**")
            st.write(
                "Positive RMSE/MAE improvement means the hybrid-trained model performed "
                "better than the real-world-only model on the test dataset. "
                "Positive R2 improvement also favors the hybrid-trained model."
            )
            render_dataframe(
                make_real_world_display_dataframe(comparison_df.round(4)), width="stretch"
            )

        st.markdown("**Test metrics**")
        test_metrics_df = metrics_df[metrics_df["split"] == "test"].sort_values(
            ["model", "training_dataset"]
        )
        test_metrics_display_df = test_metrics_df.drop(
            columns=["split", "training_seconds"],
            errors="ignore",
        )
        test_metrics_display_df = rename_percentage_metric_columns(test_metrics_display_df)
        render_dataframe(
            make_real_world_display_dataframe(test_metrics_display_df),
            width="stretch",
        )

        with st.expander("All train/validation/test metrics"):
            all_metrics_display_df = metrics_df.drop(
                columns=["training_seconds"],
                errors="ignore",
            )
            all_metrics_display_df = make_real_world_display_dataframe(all_metrics_display_df)
            all_metrics_display_df = rename_percentage_metric_columns(all_metrics_display_df)
            render_dataframe(all_metrics_display_df, width="stretch")

        if predictions_df is not None:
            with st.expander("Test predictions preview"):
                render_dataframe(
                    make_prediction_preview_display_dataframe(predictions_df.head(50)),
                    width="stretch",
                )
            prediction_columns = [
                col for col in predictions_df.columns if col.endswith(" prediction")
            ]
            if prediction_columns:
                st.markdown("""---""")
                st.markdown(":blue[**Prediction plot:**]")
                selected_prediction = st.selectbox(
                    "Prediction plot model",
                    prediction_columns,
                    format_func=lambda value: format_real_world_display_label(
                        value.replace(" prediction", "")
                    ),
                )
                plot_predictions_df = predictions_df[["y_true", selected_prediction]].dropna()
                if plot_predictions_df.empty:
                    st.warning("No valid true/predicted values are available to plot.")
                else:
                    training_thresholds = (st.session_state["training_details"] or {}).get(
                        "thresholds"
                    )
                    if training_thresholds:
                        st.caption(
                            "Point colors follow threshold classification: true/false "
                            "positive/negative based on whether true and predicted values are "
                            "inside or outside the acceptable threshold range."
                        )
                    else:
                        st.info("No thresholds are configured for this trained target column.")
                    st.altair_chart(
                        build_prediction_plot_layers(
                            plot_predictions_df,
                            selected_prediction,
                            training_thresholds,
                        ),
                        width="stretch",
                    )

            st.download_button(
                label="Download test predictions (CSV)",
                data=dataframe_to_csv_bytes(predictions_df),
                file_name="regression_test_predictions.csv",
                mime="text/csv",
            )

        st.download_button(
            label="Download regression metrics (CSV)",
            data=dataframe_to_csv_bytes(metrics_df),
            file_name="regression_model_metrics.csv",
            mime="text/csv",
        )

    if st.button("Next: Summary", disabled=metrics_df is None):
        go_to_step(7)


def render_summary_step() -> None:
    st.subheader(":blue[Step 7: Summary & Downloads]", divider="blue")
    df_exp = st.session_state["experimental_df"]
    df_sim = st.session_state["simulation_df"]
    schema = st.session_state["schema_check"]
    merged = st.session_state["merged_df"]
    metrics = st.session_state["model_metrics_df"]
    predictions = st.session_state["model_predictions_df"]
    models = st.session_state["trained_models"]
    training_details = st.session_state["training_details"]
    merge_details = st.session_state["merge_details"]

    summary = {
        "experimental_shape": list(df_exp.shape) if df_exp is not None else None,
        "simulation_shape": list(df_sim.shape) if df_sim is not None else None,
        "schema_check_run": schema is not None,
        "schema_columns_match": (not schema.has_column_mismatch) if schema is not None else None,
        "schema_dtypes_match": (not schema.has_dtype_mismatch) if schema is not None else None,
        "column_mapping_applied": st.session_state["column_mapping_applied"],
        "applied_column_mapping": st.session_state["applied_column_mapping"],
        "imbalance_summary": st.session_state["imbalance_summary"],
        "merged_shape": list(merged.shape) if merged is not None else None,
        "hybrid_analysis_completed": st.session_state["hybrid_analysis_completed"],
        "merge_details": merge_details,
        "training_details": training_details,
        "models_trained": sorted(metrics["model"].unique().tolist()) if metrics is not None else [],
    }

    st.json(summary)

    if metrics is not None:
        st.markdown("**Model metrics**")
        render_dataframe(metrics, width="stretch")

    summary_json = json.dumps(summary, indent=2).encode("utf-8")
    if summary_json is not None:
        st.markdown("**Results/artifacts download**")
        st.download_button(
            label="Download summary (JSON)",
            data=summary_json,
            file_name="training_summary.json",
            mime="application/json",
        )

    if metrics is not None:
        st.download_button(
            label="Download regression metrics (CSV)",
            data=dataframe_to_csv_bytes(metrics),
            file_name="regression_model_metrics.csv",
            mime="text/csv",
        )
    if predictions is not None:
        st.download_button(
            label="Download test predictions (CSV)",
            data=dataframe_to_csv_bytes(predictions),
            file_name="regression_test_predictions.csv",
            mime="text/csv",
        )

    if models:
        for artifact_filename, model_blob in models.items():
            st.download_button(
                label=f"Download {artifact_filename}",
                data=model_blob,
                file_name=artifact_filename,
                mime="application/octet-stream",
            )
    else:
        st.info("No trained models available yet. Train models in Step 6.")

    if st.button("Start new run"):
        st.session_state["experimental_df"] = None
        st.session_state["simulation_df"] = None
        reset_downstream_state()
        go_to_step(1)


def main() -> None:
    st.set_page_config(
        page_title="Generic Module: Hybrid Model",
        layout="wide",
    )
    init_state()
    apply_app_styles()

    scroll_to_top_once()

    st.title("Generic Module: Hybrid Model")
    unlocked_step = get_unlocked_step()
    current_step = st.session_state["current_step"]
    if current_step > unlocked_step:
        current_step = unlocked_step
        st.session_state["current_step"] = current_step

    step_items = [sac.StepsItem(title=f"{idx}. {STEP_LABELS[idx]}") for idx in range(1, 8)]
    with st.sidebar:
        logo_path = Path("./logo.png")
        if logo_path.exists():
            st.image(str(logo_path), width="stretch")
        st.markdown("**Workflow**")
        st.caption(f"Current progress: step {current_step} of {len(STEP_LABELS)}")
        selected_step_idx = sac.steps(
            items=step_items,
            index=current_step - 1,
            return_index=True,
            direction="vertical",
            size="small",
        )
    selected_step = (selected_step_idx + 1) if isinstance(selected_step_idx, int) else current_step
    if selected_step > unlocked_step:
        selected_step = unlocked_step
    if selected_step != current_step:
        st.session_state["current_step"] = selected_step
        st.session_state["scroll_to_top"] = True
        st.rerun()
    st.session_state["current_step"] = selected_step

    if selected_step == 1:
        render_upload_step()
    elif selected_step == 2:
        render_schema_step()
    elif selected_step == 3:
        render_imbalance_step()
    elif selected_step == 4:
        render_merge_step()
    elif selected_step == 5:
        render_hybrid_analysis_step()
    elif selected_step == 6:
        render_training_step()
    else:
        render_summary_step()


if __name__ == "__main__":
    main()
