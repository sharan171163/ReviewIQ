```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import pymupdf
import plotly.express as px
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent

TFIDF_PATH = PROJECT_PATH / "tfidf_vectorizer.joblib"
SENTIMENT_MODEL_PATH = PROJECT_PATH / "linear_svm_sentiment_model.joblib"

RESULTS_PATH = PROJECT_PATH / "results"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ReviewIQ — Customer Review Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found:\n{TFIDF_PATH}"
        )

    if not SENTIMENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Linear SVM model not found:\n{SENTIMENT_MODEL_PATH}"
        )

    tfidf = joblib.load(TFIDF_PATH)

    sentiment_model = joblib.load(
        SENTIMENT_MODEL_PATH
    )

    return tfidf, sentiment_model


# ============================================================
# LOAD MODEL WITH ERROR HANDLING
# ============================================================

try:

    tfidf, sentiment_model = load_model()

except Exception as e:

    st.error("❌ Unable to load the ReviewIQ model files.")

    st.code(
        str(e),
        language="text"
    )

    st.info(
        "Make sure the following files are present in the "
        "same folder as reviewiq_app.py:"
    )

    st.code(
        "tfidf_vectorizer.joblib\n"
        "linear_svm_sentiment_model.joblib",
        language="text"
    )

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SENTIMENT LABELS
# ============================================================

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def predict_sentiment(text):

    cleaned = clean_text(text)

    # Handle empty text
    if not cleaned:

        return (
            "Neutral",
            0.0,
            np.array([0.0, 0.0, 0.0])
        )

    # Convert text into TF-IDF features
    vector = tfidf.transform(
        [cleaned]
    )

    # Linear SVM prediction
    prediction = sentiment_model.predict(
        vector
    )[0]

    # --------------------------------------------------------
    # LinearSVC does not provide predict_proba().
    # Therefore, decision_function() is used.
    # --------------------------------------------------------

    decision_scores = sentiment_model.decision_function(
        vector
    )

    decision_scores = np.asarray(
        decision_scores
    ).ravel()

    # --------------------------------------------------------
    # Convert decision scores into relative scores
    # for visualization.
    #
    # NOTE:
    # These are NOT calibrated probabilities.
    # --------------------------------------------------------

    exp_scores = np.exp(
        decision_scores - np.max(decision_scores)
    )

    relative_scores = (
        exp_scores / exp_scores.sum()
    )

    prediction_int = int(prediction)

    sentiment = LABEL_MAP.get(
        prediction_int,
        str(prediction)
    )

    confidence = (
        relative_scores[prediction_int] * 100
    )

    return (
        sentiment,
        confidence,
        relative_scores
    )


# ============================================================
# FILE EXTRACTION — PDF
# ============================================================

def extract_pdf_text(file_bytes):

    text = ""

    document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page in document:

        text += page.get_text()

        text += "\n"

    document.close()

    return text


# ============================================================
# FILE EXTRACTION — TXT
# ============================================================

def extract_txt_text(file_bytes):

    return file_bytes.decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# ANALYZE REVIEW LIST
# ============================================================

def analyze_reviews(review_list):

    results = []

    total_reviews = len(
        review_list
    )

    progress_bar = st.progress(
        0
    )

    for i, text in enumerate(
        review_list
    ):

        sentiment, confidence, scores = (
            predict_sentiment(text)
        )

        results.append(
            {
                "review_text": text,
                "predicted_sentiment": sentiment,
                "sentiment_confidence": confidence,
                "negative_score": scores[0] * 100,
                "neutral_score": scores[1] * 100,
                "positive_score": scores[2] * 100
            }
        )

        if total_reviews > 0:

            progress_bar.progress(
                (i + 1) / total_reviews
            )

    progress_bar.empty()

    return pd.DataFrame(
        results
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 ReviewIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Customer Review Analytics'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "ReviewIQ"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Analyze Reviews",
        "Single Review",
        "Dashboard"
    ]
)

st.sidebar.markdown(
    "---"
)

st.sidebar.info(
    "ReviewIQ uses a trained TF-IDF + Linear SVM "
    "sentiment model to classify customer reviews."
)


# ============================================================
# SINGLE REVIEW
# ============================================================

if page == "Single Review":

    st.header(
        "🔍 Analyze a Single Review"
    )

    review_text = st.text_area(
        "Enter customer review",
        height=180,
        placeholder=(
            "Example: The product is excellent "
            "and works perfectly!"
        )
    )

    if st.button(
        "Analyze Review"
    ):

        if review_text.strip():

            sentiment, confidence, probabilities = (
                predict_sentiment(
                    review_text
                )
            )

            st.subheader(
                "Prediction"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Sentiment",
                    sentiment
                )

            with col2:

                st.metric(
                    "Relative Confidence",
                    f"{confidence:.2f}%"
                )

            # ------------------------------------------------
            # CREATE SCORE DATAFRAME
            # ------------------------------------------------

            probability_df = pd.DataFrame(
                {
                    "Sentiment": [
                        "Negative",
                        "Neutral",
                        "Positive"
                    ],
                    "Score": [
                        float(
                            probabilities[0] * 100
                        ),
                        float(
                            probabilities[1] * 100
                        ),
                        float(
                            probabilities[2] * 100
                        )
                    ]
                }
            )

            # ------------------------------------------------
            # SENTIMENT SCORE CHART
            # ------------------------------------------------

            st.subheader(
                "📊 Sentiment Scores"
            )

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Score",
                text="Score",
                title="Relative Sentiment Scores"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Relative Score (%)",
                xaxis_title="",
                yaxis_range=[
                    0,
                    100
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Note: Linear SVM does not produce calibrated "
                "probabilities. The displayed values are "
                "relative scores derived from the model's "
                "decision function."
            )

        else:

            st.warning(
                "Please enter a review."
            )


# ============================================================
# FILE ANALYSIS
# ============================================================

elif page == "Analyze Reviews":

    st.header(
        "📁 Analyze Review File"
    )

    st.write(
        "Upload a CSV, PDF, or TXT file containing "
        "customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload Review File",
        type=[
            "csv",
            "pdf",
            "txt"
        ]
    )

    if uploaded_file is not None:

        file_name = (
            uploaded_file.name.lower()
        )

        # ====================================================
        # CSV ANALYSIS
        # ====================================================

        if file_name.endswith(
            ".csv"
        ):

            try:

                df = pd.read_csv(
                    uploaded_file,
                    encoding="utf-8"
                )

            except Exception:

                uploaded_file.seek(
                    0
                )

                try:

                    df = pd.read_csv(
                        uploaded_file,
                        encoding="cp1252"
                    )

                except Exception as e:

                    st.error(
                        "Unable to read the CSV file."
                    )

                    st.code(
                        str(e),
                        language="text"
                    )

                    st.stop()

            if df.empty:

                st.warning(
                    "The uploaded CSV file is empty."
                )

            else:

                st.success(
                    f"CSV loaded successfully — "
                    f"{len(df):,} rows"
                )

                st.subheader(
                    "Preview"
                )

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                # ------------------------------------------------
                # FIND TEXT COLUMNS
                # ------------------------------------------------

                text_columns = [
                    col
                    for col in df.columns
                    if (
                        df[col].dtype == "object"
                        or pd.api.types.is_string_dtype(
                            df[col]
                        )
                    )
                ]

                if text_columns:

                    selected_column = st.selectbox(
                        "Select the review text column",
                        text_columns
                    )

                    if st.button(
                        "🔍 Analyze CSV",
                        key="analyze_csv"
                    ):

                        working_df = (
                            df.copy()
                        )

                        working_df[
                            "review_text"
                        ] = (
                            working_df[
                                selected_column
                            ]
                            .fillna("")
                            .astype(str)
                        )

                        review_list = (
                            working_df[
                                "review_text"
                            ]
                            .tolist()
                        )

                        output_results = (
                            analyze_reviews(
                                review_list
                            )
                        )

                        # ------------------------------------------------
                        # COMBINE ORIGINAL DATA + PREDICTIONS
                        # ------------------------------------------------

                        output_df = pd.concat(
                            [
                                working_df.reset_index(
                                    drop=True
                                ),
                                output_results[
                                    [
                                        "predicted_sentiment",
                                        "sentiment_confidence",
                                        "negative_score",
                                        "neutral_score",
                                        "positive_score"
                                    ]
                                ].reset_index(
                                    drop=True
                                )
                            ],
                            axis=1
                        )

                        st.session_state[
                            "analysis_df"
                        ] = output_df

                        st.session_state[
                            "analysis_source"
                        ] = "CSV"

                        st.success(
                            "✅ CSV analysis completed successfully!"
                        )

                        st.info(
                            "Go to **Dashboard** to view "
                            "the analysis."
                        )

                else:

                    st.warning(
                        "No text column was found in the CSV."
                    )

                    st.info(
                        "Your CSV should contain a column "
                        "containing customer review text."
                    )


        # ====================================================
        # PDF ANALYSIS
        # ====================================================

        elif file_name.endswith(
            ".pdf"
        ):

            file_bytes = (
                uploaded_file.read()
            )

            try:

                extracted_text = (
                    extract_pdf_text(
                        file_bytes
                    )
                )

            except Exception as e:

                st.error(
                    "Unable to extract text from the PDF."
                )

                st.code(
                    str(e),
                    language="text"
                )

                st.stop()

            if not extracted_text.strip():

                st.warning(
                    "No text could be extracted from this PDF."
                )

            else:

                st.success(
                    "PDF text extracted successfully!"
                )

                st.text_area(
                    "Extracted Text Preview",
                    extracted_text[
                        :5000
                    ],
                    height=250
                )

                st.write(
                    "Each non-empty line will be treated "
                    "as a separate review."
                )

                if st.button(
                    "🔍 Analyze PDF",
                    key="analyze_pdf"
                ):

                    # ------------------------------------------------
                    # SPLIT PDF INTO REVIEWS
                    # ------------------------------------------------

                    review_lines = [
                        line.strip()
                        for line
                        in extracted_text.splitlines()
                        if line.strip()
                    ]

                    # Remove extremely short lines
                    review_lines = [
                        line
                        for line
                        in review_lines
                        if len(line) >= 10
                    ]

                    if not review_lines:

                        st.warning(
                            "No review text could be detected "
                            "in the PDF."
                        )

                    else:

                        output_df = (
                            analyze_reviews(
                                review_lines
                            )
                        )

                        st.session_state[
                            "analysis_df"
                        ] = output_df

                        st.session_state[
                            "analysis_source"
                        ] = "PDF"

                        st.success(
                            f"✅ PDF analysis completed — "
                            f"{len(output_df):,} reviews analyzed."
                        )

                        st.info(
                            "Go to **Dashboard** to view "
                            "the analysis."
                        )


        # ====================================================
        # TXT ANALYSIS
        # ====================================================

        elif file_name.endswith(
            ".txt"
        ):

            file_bytes = (
                uploaded_file.read()
            )

            extracted_text = (
                extract_txt_text(
                    file_bytes
                )
            )

            if not extracted_text.strip():

                st.warning(
                    "The TXT file is empty."
                )

            else:

                st.success(
                    "TXT file loaded successfully!"
                )

                st.text_area(
                    "Review Text Preview",
                    extracted_text[
                        :5000
                    ],
                    height=250
                )

                st.write(
                    "Each non-empty line will be treated "
                    "as a separate review."
                )

                if st.button(
                    "🔍 Analyze TXT",
                    key="analyze_txt"
                ):

                    # ------------------------------------------------
                    # SPLIT TXT INTO REVIEWS
                    # ------------------------------------------------

                    review_lines = [
                        line.strip()
                        for line
                        in extracted_text.splitlines()
                        if line.strip()
                    ]

                    review_lines = [
                        line
                        for line
                        in review_lines
                        if len(line) >= 10
                    ]

                    if not review_lines:

                        st.warning(
                            "No review text could be detected "
                            "in the TXT file."
                        )

                    else:

                        output_df = (
                            analyze_reviews(
                                review_lines
                            )
                        )

                        st.session_state[
                            "analysis_df"
                        ] = output_df

                        st.session_state[
                            "analysis_source"
                        ] = "TXT"

                        st.success(
                            f"✅ TXT analysis completed — "
                            f"{len(output_df):,} reviews analyzed."
                        )

                        st.info(
                            "Go to **Dashboard** to view "
                            "the analysis."
                        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.header(
        "📊 ReviewIQ Dashboard"
    )

    # --------------------------------------------------------
    # CHECK FOR ANALYZED DATA
    # --------------------------------------------------------

    if (
        "analysis_df"
        not in st.session_state
    ):

        st.info(
            "📁 No analyzed review data available yet."
        )

        st.write(
            "Go to **Analyze Reviews**, upload a CSV, PDF, "
            "or TXT file, analyze it, and then return "
            "to the Dashboard."
        )

    else:

        # ----------------------------------------------------
        # LOAD ANALYZED DATA
        # ----------------------------------------------------

        dashboard_df = (
            st.session_state[
                "analysis_df"
            ].copy()
        )

        analysis_source = (
            st.session_state.get(
                "analysis_source",
                "Uploaded File"
            )
        )

        # ----------------------------------------------------
        # NORMALIZE SENTIMENT LABELS
        # ----------------------------------------------------

        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        }

        if (
            "predicted_sentiment"
            in dashboard_df.columns
        ):

            dashboard_df[
                "predicted_sentiment"
            ] = (
                dashboard_df[
                    "predicted_sentiment"
                ]
                .map(label_map)
                .fillna(
                    dashboard_df[
                        "predicted_sentiment"
                    ]
                )
            )

        # ----------------------------------------------------
        # DASHBOARD STATUS
        # ----------------------------------------------------

        st.success(
            f"Dashboard showing your analyzed "
            f"{analysis_source} data — "
            f"{len(dashboard_df):,} reviews"
        )

        # ----------------------------------------------------
        # CHECK SENTIMENT COLUMN
        # ----------------------------------------------------

        if (
            "predicted_sentiment"
            not in dashboard_df.columns
        ):

            st.error(
                "Sentiment prediction data is missing."
            )

            st.stop()

        # ----------------------------------------------------
        # SENTIMENT COUNTS
        # ----------------------------------------------------

        sentiment_counts = (
            dashboard_df[
                "predicted_sentiment"
            ]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "sentiment",
            "review_count"
        ]

        # ----------------------------------------------------
        # TOTALS
        # ----------------------------------------------------

        total = len(
            dashboard_df
        )

        positive = int(
            sentiment_counts.loc[
                sentiment_counts[
                    "sentiment"
                ] == "Positive",
                "review_count"
            ].sum()
        )

        negative = int(
            sentiment_counts.loc[
                sentiment_counts[
                    "sentiment"
                ] == "Negative",
                "review_count"
            ].sum()
        )

        neutral = int(
            sentiment_counts.loc[
                sentiment_counts[
                    "sentiment"
                ] == "Neutral",
                "review_count"
            ].sum()
        )

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Total Reviews",
                f"{total:,}"
            )

        with col2:

            positive_percentage = (
                positive / total * 100
                if total > 0
                else 0
            )

            st.metric(
                "Positive",
                f"{positive_percentage:.2f}%"
            )

        with col3:

            negative_percentage = (
                negative / total * 100
                if total > 0
                else 0
            )

            st.metric(
                "Negative",
                f"{negative_percentage:.2f}%"
            )

        with col4:

            neutral_percentage = (
                neutral / total * 100
                if total > 0
                else 0
            )

            st.metric(
                "Neutral",
                f"{neutral_percentage:.2f}%"
            )

        # ----------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "📊 Sentiment Distribution"
        )

        fig = px.bar(
            sentiment_counts,
            x="sentiment",
            y="review_count",
            text="review_count",
            title="Customer Review Sentiment"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Number of Reviews"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SENTIMENT PERCENTAGE
        # ----------------------------------------------------

        st.subheader(
            "🥧 Sentiment Percentage"
        )

        percentage_df = (
            sentiment_counts.copy()
        )

        percentage_df[
            "percentage"
        ] = (
            percentage_df[
                "review_count"
            ]
            / total
            * 100
            if total > 0
            else 0
        )

        fig2 = px.pie(
            percentage_df,
            names="sentiment",
            values="review_count",
            hole=0.45,
            title="Sentiment Percentage"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ----------------------------------------------------
        # SENTIMENT SCORE DISTRIBUTION
        # ----------------------------------------------------

        score_columns = [
            "negative_score",
            "neutral_score",
            "positive_score"
        ]

        if all(
            column in dashboard_df.columns
            for column in score_columns
        ):

            st.subheader(
                "📈 Average Sentiment Scores"
            )

            average_scores = pd.DataFrame(
                {
                    "Sentiment": [
                        "Negative",
                        "Neutral",
                        "Positive"
                    ],
                    "Average Score": [
                        dashboard_df[
                            "negative_score"
                        ].mean(),
                        dashboard_df[
                            "neutral_score"
                        ].mean(),
                        dashboard_df[
                            "positive_score"
                        ].mean()
                    ]
                }
            )

            score_fig = px.bar(
                average_scores,
                x="Sentiment",
                y="Average Score",
                text="Average Score",
                title="Average Model Sentiment Scores"
            )

            score_fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            score_fig.update_layout(
                yaxis_title="Average Relative Score (%)",
                xaxis_title="",
                yaxis_range=[
                    0,
                    100
                ]
            )

            st.plotly_chart(
                score_fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # ANALYZED REVIEW DATA
        # ----------------------------------------------------

        st.subheader(
            "📋 Analyzed Reviews"
        )

        st.dataframe(
            dashboard_df,
            use_container_width=True,
            height=400
        )

        # ----------------------------------------------------
        # DOWNLOAD ANALYZED DATA
        # ----------------------------------------------------

        csv_data = (
            dashboard_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Analyzed Reviews",
            data=csv_data,
            file_name=(
                "reviewiq_analyzed_reviews.csv"
            ),
            mime="text/csv"
        )

        # ----------------------------------------------------
        # CLEAR ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            "---"
        )

        if st.button(
            "🗑️ Clear Current Analysis"
        ):

            if (
                "analysis_df"
                in st.session_state
            ):

                del st.session_state[
                    "analysis_df"
                ]

            if (
                "analysis_source"
                in st.session_state
            ):

                del st.session_state[
                    "analysis_source"
                ]

            st.success(
                "Current analysis cleared."
            )

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "ReviewIQ | Customer Review Analytics | "
    "TF-IDF + Linear SVM"
)
