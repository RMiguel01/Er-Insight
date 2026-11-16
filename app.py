
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

from xray_tab import show_xray_tab


# ============================================================
# APP CONFIGURATION AND DESIGN
# ============================================================

st.set_page_config(
    page_title="ER Insight",
    page_icon="🏥",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #172B4D;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.6rem;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DBE5F1;
        border-radius: 12px;
        padding: 14px 18px;
    }

    div[data-testid="stMetricValue"] {
        color: #1D4ED8;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563EB;
        border-color: #2563EB;
        color: #FFFFFF;
        border-radius: 8px;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }

    div[data-testid="stSelectboxVirtualDropdown"]
    [role="option"][aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"]
    [role="option"][aria-selected="true"] * {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "admission_model.joblib"

ECG_DATA_FILE = (
    BASE_DIR / "data" / "ptbxl" / "ptbxl_database.csv"
)

ECG_CODES_FILE = (
    BASE_DIR / "data" / "ptbxl" / "scp_statements.csv"
)

NOTES_FILE = BASE_DIR / "data" / "ecg_notes.json"


# ============================================================
# YALE MODEL FEATURES
# ============================================================

YALE_FEATURES = [
    "age",
    "esi",
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_temp"
]


# ============================================================
# ECG LEAD DESCRIPTIONS
# ============================================================

LEAD_DESCRIPTIONS = {
    "I": "Lateral view",
    "II": "Inferior view - commonly used for rhythm",
    "III": "Inferior view",
    "aVR": "Right-sided augmented view",
    "aVL": "High lateral augmented view",
    "aVF": "Inferior augmented view",
    "V1": "Right precordial view",
    "V2": "Septal precordial view",
    "V3": "Anterior precordial view",
    "V4": "Anterior precordial view",
    "V5": "Lateral precordial view",
    "V6": "Lateral precordial view"
}


# ============================================================
# ECG FINDING EXPLANATIONS
# ============================================================

EXPLANATIONS = {
    "NORM": (
        "This ECG is labeled normal in the "
        "original research dataset."
    ),
    "SR": (
        "Sinus rhythm: the heartbeat originates "
        "from the heart's natural pacemaker."
    ),
    "SBRAD": (
        "Sinus bradycardia: a slower-than-usual "
        "sinus rhythm."
    ),
    "STACH": (
        "Sinus tachycardia: a faster-than-usual "
        "sinus rhythm."
    ),
    "AFIB": (
        "Atrial fibrillation: an irregular heart "
        "rhythm originating in the atria."
    ),
    "PVC": (
        "Premature ventricular contractions: "
        "extra beats originating in the ventricles."
    ),
    "LBBB": (
        "Left bundle branch block: delayed "
        "electrical conduction in the left "
        "bundle branch."
    ),
    "RBBB": (
        "Right bundle branch block: delayed "
        "electrical conduction in the right "
        "bundle branch."
    )
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_ecg_data():

    ecg_data = pd.read_csv(ECG_DATA_FILE)

    scp_codes = pd.read_csv(
        ECG_CODES_FILE,
        index_col=0
    )

    return ecg_data, scp_codes


@st.cache_data(show_spinner=False, max_entries=20)
def load_ecg_waveform(filename):

    filename = str(filename).replace("\\", "/")

    folder, record_name = filename.rsplit("/", 1)

    recording = wfdb.rdrecord(
        record_name,
        pn_dir=f"ptb-xl/1.0.3/{folder}"
    )

    return (
        recording.p_signal,
        recording.fs,
        recording.sig_name
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_codes(value):

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, dict):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return {}


def ecg_age_text(value):

    try:

        age = float(value)

        if np.isnan(age):
            return "Unknown"

        if age >= 300:
            return "90+"

        return str(int(age))

    except (ValueError, TypeError):
        return "Unknown"


def code_description(code, statements):

    if code not in statements.index:
        return code

    description = statements.loc[
        code,
        "description"
    ]

    if isinstance(description, pd.Series):
        description = description.iloc[0]

    if pd.isna(description):
        return code

    description = str(description).strip()

    if not description:
        return code

    return description


# ============================================================
# PATIENT NOTES
# ============================================================

def load_saved_notes():

    try:

        if NOTES_FILE.exists():

            with NOTES_FILE.open(
                "r",
                encoding="utf-8"
            ) as file:

                notes = json.load(file)

                if isinstance(notes, dict):
                    return notes

    except (OSError, json.JSONDecodeError):
        pass

    return {}


def save_patient_notes(notes):

    NOTES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with NOTES_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            notes,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# WEBSITE HEADER
# ============================================================

st.title("🏥 ER Insight")

st.markdown(
    "**Emergency Department Admission Risk "
    "& Clinical Research Explorer**"
)

st.caption(
    "An interactive research prototype combining "
    "emergency admission prediction with independent "
    "ECG and chest X-ray research collections."
)

st.info(
    "Research and educational demonstration only. "
    "Not clinically validated. Use fictional patient "
    "scenarios and do not enter real patient information."
)


# ============================================================
# MAIN NAVIGATION
# ============================================================

tab_yale, tab_ecg, tab_xray = st.tabs([
    "🏥 Yale Admission Prediction",
    "❤️ PTB-XL ECG Records",
    "🩻 NIH Chest X-rays"
])


# ============================================================
# TAB 1: YALE ADMISSION PREDICTION
# ============================================================

with tab_yale:

    st.write(
        "Enter fictional triage information to explore "
        "how the research model estimates the probability "
        "of hospital admission."
    )

    st.caption(
        "These input ranges are prototype limits, "
        "not clinical reference ranges. "
        "ESI 1 represents the highest acuity."
    )

    st.subheader("Patient Triage Information")

    left, right = st.columns(2, gap="large")

    # --------------------------------------
    # LEFT COLUMN
    # --------------------------------------

    with left:

        patient_age = st.number_input(
            "Patient Age (0–110 years)",
            min_value=0,
            max_value=110,
            value=50
        )

        esi = st.selectbox(
            "Emergency Severity Index (ESI: 1–5)",
            options=[1, 2, 3, 4, 5],
            index=2
        )

        hr = st.number_input(
            "Heart Rate (20–250 bpm)",
            min_value=20,
            max_value=250,
            value=80
        )

        sbp = st.number_input(
            "Systolic Blood Pressure (40–300 mmHg)",
            min_value=40,
            max_value=300,
            value=120
        )

    # --------------------------------------
    # RIGHT COLUMN
    # --------------------------------------

    with right:

        dbp = st.number_input(
            "Diastolic Blood Pressure (20–200 mmHg)",
            min_value=20,
            max_value=200,
            value=80
        )

        rr = st.number_input(
            "Respiratory Rate (4–80 breaths/min)",
            min_value=4,
            max_value=80,
            value=16
        )

        o2 = st.number_input(
            "Oxygen Saturation (50–100%)",
            min_value=50,
            max_value=100,
            value=98
        )

        temp = st.number_input(
            "Body Temperature (85–110°F)",
            min_value=85.0,
            max_value=110.0,
            value=98.6,
            step=0.1
        )

    # --------------------------------------
    # ADMISSION PREDICTION
    # --------------------------------------

    if st.button(
        "Analyze Admission Risk",
        type="primary",
        use_container_width=True
    ):

        patient_data = pd.DataFrame(
            [
                {
                    "age": patient_age,
                    "esi": esi,
                    "triage_vital_hr": hr,
                    "triage_vital_sbp": sbp,
                    "triage_vital_dbp": dbp,
                    "triage_vital_rr": rr,
                    "triage_vital_o2": o2,
                    "triage_vital_temp": temp
                }
            ],
            columns=YALE_FEATURES
        )

        try:

            model = load_model()

            probability = float(
                model.predict_proba(
                    patient_data
                )[0][1]
            )

            st.session_state[
                "yale_probability"
            ] = probability

            st.session_state[
                "yale_last_inputs"
            ] = patient_data.iloc[0].to_dict()

        except Exception as error:

            st.error(
                f"Unable to calculate admission risk: "
                f"{error}"
            )

    # --------------------------------------
    # ADMISSION RISK RESULT
    # --------------------------------------

    if "yale_probability" in st.session_state:

        probability = st.session_state[
            "yale_probability"
        ]

        st.divider()

        st.subheader("Admission Risk Estimate")

        st.metric(
            "Estimated Probability of Admission",
            f"{probability:.1%}"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.caption(
            "Logistic Regression | Yale dataset | "
            "Held-out ROC-AUC: 0.8346 "
            "(random visit-level split)"
        )

        current_inputs = {
            "age": patient_age,
            "esi": esi,
            "triage_vital_hr": hr,
            "triage_vital_sbp": sbp,
            "triage_vital_dbp": dbp,
            "triage_vital_rr": rr,
            "triage_vital_o2": o2,
            "triage_vital_temp": temp
        }

        if current_inputs != st.session_state.get(
            "yale_last_inputs"
        ):

            st.info(
                "Triage values have changed. "
                "Click Analyze Admission Risk again "
                "to update the estimate."
            )

        st.warning(
            "This research estimate is not a medical "
            "recommendation or an admission decision. "
            "The model has not been clinically validated."
        )

    # --------------------------------------
    # ABOUT ER INSIGHT
    # --------------------------------------

    st.divider()

    st.subheader("About ER Insight")

    st.write(
        "ER Insight combines a Yale-based emergency "
        "admission-risk research model with separate "
        "ECG and chest X-ray research explorers."
    )

    st.write(
        "The Yale, PTB-XL, and NIH datasets contain "
        "independent research records. Their patients "
        "are not linked, and the ECG and X-ray findings "
        "do not influence the Yale prediction."
    )

    st.caption(
        "Research and educational use only. "
        "Not intended for real patient care."
    )


# ============================================================
# TAB 2: PTB-XL ECG EXPLORER
# ============================================================

with tab_ecg:

    st.header("PTB-XL ECG Explorer")

    st.write(
        "Explore anonymized ECG research records, "
        "existing findings, and actual ECG waveforms."
    )

    try:

        ecg_data, scp_codes = load_ecg_data()

        st.metric(
            "ECG Records in Dataset",
            f"{len(ecg_data):,}"
        )

        st.caption(
            "The dropdown displays the first "
            "100 ECG records."
        )

        # ----------------------------------
        # ECG PATIENT SELECTION
        # ----------------------------------

        ecg_options = {}

        for _, row in ecg_data.head(100).iterrows():

            ecg_id = int(row["ecg_id"])
            patient_id = int(row["patient_id"])

            ecg_options[ecg_id] = (
                f"Patient #{patient_id} | "
                f"Age {ecg_age_text(row['age'])} | "
                f"ECG Record #{ecg_id}"
            )

        selected_ecg = st.selectbox(
            "Select an ECG Patient Record",
            options=list(ecg_options.keys()),
            format_func=lambda value: ecg_options[value],
            key="selected_ecg_record"
        )

        record = ecg_data.loc[
            ecg_data["ecg_id"] == selected_ecg
        ].iloc[0]

        patient_id = int(
            record["patient_id"]
        )

        record_key = str(selected_ecg)

        # ----------------------------------
        # PATIENT INFORMATION
        # ----------------------------------

        st.divider()

        st.subheader("Patient Information")

        info1, info2, info3 = st.columns(3)

        info1.metric(
            "ECG Record ID",
            str(selected_ecg)
        )

        info2.metric(
            "Patient Age",
            ecg_age_text(record["age"])
        )

        sex_value = record.get("sex")

        if sex_value == 1:
            recorded_sex = "Female"

        elif sex_value == 0:
            recorded_sex = "Male"

        else:
            recorded_sex = "Unknown"

        info3.metric(
            "Recorded Sex",
            recorded_sex
        )

        st.write(
            f"**De-identified Patient ID:** {patient_id}"
        )

        # ==================================
        # ECG FINDINGS
        # ==================================

        st.divider()

        st.subheader("ECG Findings")

        st.caption(
            "Original findings from the PTB-XL "
            "research dataset. These are not "
            "diagnoses generated by ER Insight."
        )

        codes = parse_codes(
            record["scp_codes"]
        )

        # ----------------------------------
        # ORIGINAL DATASET FINDINGS
        # ----------------------------------

        if codes:

            st.write(
                f"**{len(codes)} original findings**"
            )

            for code in codes:

                description = code_description(
                    code,
                    scp_codes
                )

                explanation = EXPLANATIONS.get(
                    code,
                    "This finding is recorded in "
                    "the original PTB-XL dataset."
                )

                with st.container(border=True):

                    st.markdown(
                        f"**{description}**"
                    )

                    st.write(explanation)

                    st.caption(
                        f"Code: {code} | "
                        "Original dataset finding"
                    )

        else:

            st.info(
                "No original diagnostic annotations "
                "are available for this ECG record."
            )

        # ==================================
        # USER-ADDED FINDINGS
        # ==================================

        if "custom_ecg_findings" not in st.session_state:

            st.session_state[
                "custom_ecg_findings"
            ] = {}

        custom_findings_by_record = st.session_state[
            "custom_ecg_findings"
        ]

        if record_key not in custom_findings_by_record:

            custom_findings_by_record[
                record_key
            ] = []

        custom_findings = custom_findings_by_record[
            record_key
        ]

        if custom_findings:

            st.markdown("**User-Added Findings**")

            for index, finding in enumerate(
                custom_findings
            ):

                with st.container(border=True):

                    st.markdown(
                        f"**{finding['title']}**"
                    )

                    st.write(
                        finding["description"]
                    )

                    st.caption(
                        "User-added demo finding | "
                        "Not part of the original "
                        "PTB-XL dataset"
                    )

                    if st.button(
                        "Remove Finding",
                        key=(
                            f"remove_ecg_finding_"
                            f"{record_key}_{index}"
                        )
                    ):

                        custom_findings.pop(index)

                        st.rerun()

        # ==================================
        # ADD CUSTOM FINDING
        # ==================================

        with st.expander("➕ Add Custom Finding"):

            st.info(
                "Demo feature only. Add a fictional "
                "finding or observation. This does "
                "not modify the original ECG data."
            )

            with st.form(
                key=f"add_ecg_finding_{record_key}",
                clear_on_submit=True
            ):

                finding_title = st.text_input(
                    "Finding Title",
                    placeholder=(
                        "Example: Follow-up Recommended"
                    )
                )

                finding_description = st.text_area(
                    "Finding Description",
                    placeholder=(
                        "Example: Additional review "
                        "suggested for this fictional "
                        "patient scenario."
                    ),
                    height=120
                )

                submitted = st.form_submit_button(
                    "Add Finding",
                    type="primary"
                )

                if submitted:

                    title = finding_title.strip()

                    description = (
                        finding_description.strip()
                    )

                    if not title or not description:

                        st.error(
                            "Please enter both a title "
                            "and a description."
                        )

                    else:

                        custom_findings.append(
                            {
                                "title": title,
                                "description": description
                            }
                        )

                        st.rerun()

        # ==================================
        # ORIGINAL ECG REPORT
        # ==================================

        with st.expander(
            "View Original ECG Report"
        ):

            st.write(
                str(
                    record.get(
                        "report",
                        "Not available"
                    )
                )
            )

            st.caption(
                "Original report text from the "
                "PTB-XL dataset. Some reports "
                "are written in German."
            )

        # ==================================
        # PATIENT NOTES
        # ==================================

        st.divider()

        st.subheader("Patient Notes")

        st.write(
            "Record fictional observations "
            "about this ECG research example."
        )

        st.info(
            "Demo-only notes. Local file storage "
            "is not secure medical-record storage. "
            "Do not enter real patient information."
        )

        notes_key = (
            f"{patient_id}_{int(selected_ecg)}"
        )

        saved_notes = load_saved_notes()

        existing_notes = saved_notes.get(
            notes_key,
            ""
        )

        user_notes = st.text_area(
            "Additional Clinical Notes",
            value=existing_notes,
            height=160,
            placeholder=(
                "Example: Fictional patient reports "
                "occasional chest discomfort."
            ),
            key=f"ecg_notes_{notes_key}"
        )

        if st.button(
            "Save Notes",
            key=f"save_ecg_notes_{notes_key}"
        ):

            saved_notes[notes_key] = user_notes

            try:

                save_patient_notes(
                    saved_notes
                )

                st.success(
                    "Demo notes saved locally."
                )

            except OSError as error:

                st.error(
                    f"Unable to save notes: {error}"
                )

        # ==================================
        # ACTUAL ECG WAVEFORM
        # ==================================

        st.divider()

        st.subheader("Actual ECG Recording")

        selected_lead = st.selectbox(
            "Select ECG Lead",
            options=list(
                LEAD_DESCRIPTIONS.keys()
            ),
            index=1,
            format_func=lambda lead: (
                f"{lead} — "
                f"{LEAD_DESCRIPTIONS[lead]}"
            )
        )

        if st.button(
            "Show ECG Waveform",
            type="primary"
        ):

            try:

                filename = str(
                    record["filename_lr"]
                )

                signals, sample_rate, lead_names = (
                    load_ecg_waveform(filename)
                )

                # PTB-XL uses uppercase AVR/AVL/AVF.
                # Normalize the names for matching.

                normalized_leads = [
                    str(lead).upper()
                    for lead in lead_names
                ]

                selected_index = normalized_leads.index(
                    selected_lead.upper()
                )

                waveform = pd.DataFrame(
                    {
                        "Time (seconds)": (
                            np.arange(
                                signals.shape[0]
                            ) / sample_rate
                        ),
                        "Amplitude (mV)": (
                            signals[:, selected_index]
                        )
                    }
                )

                st.line_chart(
                    waveform,
                    x="Time (seconds)",
                    y="Amplitude (mV)"
                )

                st.caption(
                    f"Original PTB-XL recording | "
                    f"Lead {selected_lead} | "
                    f"Sampling rate: "
                    f"{sample_rate:g} Hz"
                )

            except ValueError:

                st.error(
                    "The selected ECG lead "
                    "is unavailable."
                )

            except Exception as error:

                st.error(
                    f"Unable to load ECG waveform: "
                    f"{error}"
                )

    except Exception as error:

        st.error(
            f"Unable to load ECG Explorer: {error}"
        )


# ============================================================
# TAB 3: NIH CHEST X-RAY EXPLORER
# ============================================================

with tab_xray:

    show_xray_tab()
