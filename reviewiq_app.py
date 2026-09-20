
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
SENTIMENT_MODEL_PATH = PROJECT_PATH / "sentiment_model.joblib"

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

st.markdown("""
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

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background: #f7f7f7;
    text-align: center;
    border: 1px solid #e5e5e5;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
}

.metric-label {
    font-size: 15px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    tfidf = joblib.load(TFIDF_PATH)
    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    return tfidf, sentiment_model

tfidf, sentiment_model = load_model()

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def predict_sentiment(text):

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = sentiment_model.predict(vector)[0]

    probabilities = sentiment_model.predict_proba(vector)[0]

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    sentiment = label_map[prediction]

    confidence = probabilities[prediction] * 100

    return sentiment, confidence, probabilities


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    text = ""

    document = pymupdf.open(stream=file_bytes, filetype="pdf")

    for page in document:
        text += page.get_text()

    document.close()

    return text


def extract_txt_text(file_bytes):

    return file_bytes.decode("utf-8", errors="ignore")


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

st.sidebar.title("ReviewIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Analyze Reviews",
        "Single Review",
        "Dashboard"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "ReviewIQ uses a trained TF-IDF + Logistic Regression "
    "sentiment model to classify customer reviews."
)

# ============================================================
# SINGLE REVIEW
# ============================================================

if page == "Single Review":

    st.header("🔍 Analyze a Single Review")

    review_text = st.text_area(
        "Enter customer review",
        height=180,
        placeholder="Example: The product is excellent and works perfectly!"
    )

    if st.button("Analyze Review"):

        if review_text.strip():

            sentiment, confidence, probabilities = predict_sentiment(
                review_text
            )

            st.subheader("Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Sentiment",
                    sentiment
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            probability_df = pd.DataFrame({
                "Sentiment": [
                    "Negative",
                    "Neutral",
                    "Positive"
                ],
                "Probability": probabilities * 100
            })

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                text="Probability",
                title="Sentiment Probabilities"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning("Please enter a review.")

# ============================================================
# FILE ANALYSIS
# ============================================================

elif page == "Analyze Reviews":

    st.header("📁 Analyze Review File")

    st.write(
        "Upload a CSV, PDF, or TXT file containing customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload Review File",
        type=["csv", "pdf", "txt"]
    )

    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if file_name.endswith(".csv"):

            try:

                df = pd.read_csv(
                    uploaded_file,
                    encoding="utf-8"
                )

            except:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="cp1252"
                )

            st.success(
                f"CSV loaded successfully — {len(df):,} rows"
            )

            st.subheader("Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            text_columns = [
                col for col in df.columns
                if df[col].dtype == "object"
            ]

            if text_columns:

                selected_column = st.selectbox(
                    "Select the review text column",
                    text_columns
                )

                if st.button("Analyze CSV"):

                    working_df = df.copy()

                    working_df["clean_text"] = (
                        working_df[selected_column]
                        .fillna("")
                        .astype(str)
                    )

                    results = []

                    for text in working_df["clean_text"]:

                        sentiment, confidence, probabilities = \
                            predict_sentiment(text)

                        results.append({
                            "predicted_sentiment": sentiment,
                            "sentiment_confidence": confidence,
                            "negative_probability":
                                probabilities[0] * 100,
                            "neutral_probability":
                                probabilities[1] * 100,
                            "positive_probability":
                                probabilities[2] * 100
                        })

                    results_df = pd.DataFrame(results)

                    output_df = pd.concat(
                        [
                            working_df.reset_index(drop=True),
                            results_df
                        ],
                        axis=1
                    )

                    st.session_state["analysis_df"] = output_df

                    st.success(
                        "CSV analysis completed successfully!"
                    )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        elif file_name.endswith(".pdf"):

            file_bytes = uploaded_file.read()

            extracted_text = extract_pdf_text(file_bytes)

            st.success(
                "PDF text extracted successfully!"
            )

            st.text_area(
                "Extracted Text",
                extracted_text[:5000],
                height=250
            )

            if st.button("Analyze PDF"):

                sentiment, confidence, probabilities = \
                    predict_sentiment(extracted_text)

                st.subheader("PDF Sentiment")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Sentiment",
                        sentiment
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        elif file_name.endswith(".txt"):

            file_bytes = uploaded_file.read()

            extracted_text = extract_txt_text(file_bytes)

            st.success(
                "TXT file loaded successfully!"
            )

            st.text_area(
                "Review Text",
                extracted_text[:5000],
                height=250
            )

            if st.button("Analyze TXT"):

                sentiment, confidence, probabilities = \
                    predict_sentiment(extracted_text)

                st.subheader("TXT Sentiment")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Sentiment",
                        sentiment
                    )

                with col2:
                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.header("📊 ReviewIQ Dashboard")

    # --------------------------------------------------------
    # CHECK FOR ANALYZED UPLOADED DATA
    # --------------------------------------------------------

    if "analysis_df" not in st.session_state:

        st.info(
            "📁 No analyzed review data available yet."
        )

        st.write(
            "Go to **Analyze Reviews**, upload a CSV file, "
            "select the review text column, and click "
            "**Analyze CSV**."
        )

    else:

        # ----------------------------------------------------
        # LOAD THE USER'S UPLOADED DATA
        # ----------------------------------------------------

        dashboard_df = st.session_state["analysis_df"].copy()

        # ----------------------------------------------------
        # CONVERT SENTIMENT LABELS
        # ----------------------------------------------------

        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        }

        dashboard_df["predicted_sentiment"] = (
            dashboard_df["predicted_sentiment"]
            .map(label_map)
            .fillna(dashboard_df["predicted_sentiment"])
        )

        # ----------------------------------------------------
        # DASHBOARD STATUS
        # ----------------------------------------------------

        st.success(
            f"Dashboard showing your uploaded data — "
            f"{len(dashboard_df):,} reviews"
        )

        # ----------------------------------------------------
        # SENTIMENT COUNTS
        # ----------------------------------------------------

        sentiment_counts = (
            dashboard_df["predicted_sentiment"]
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

        total = len(dashboard_df)

        positive = (
            sentiment_counts.loc[
                sentiment_counts["sentiment"] == "Positive",
                "review_count"
            ].sum()
        )

        negative = (
            sentiment_counts.loc[
                sentiment_counts["sentiment"] == "Negative",
                "review_count"
            ].sum()
        )

        neutral = (
            sentiment_counts.loc[
                sentiment_counts["sentiment"] == "Neutral",
                "review_count"
            ].sum()
        )

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Reviews",
                f"{total:,}"
            )

        with col2:

            st.metric(
                "Positive",
                f"{positive / total * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Negative",
                f"{negative / total * 100:.2f}%"
            )

        with col4:

            st.metric(
                "Neutral",
                f"{neutral / total * 100:.2f}%"
            )

        # ----------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("📊 Sentiment Distribution")

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

        st.subheader("🥧 Sentiment Percentage")

        percentage_df = sentiment_counts.copy()

        percentage_df["percentage"] = (
            percentage_df["review_count"]
            / total
            * 100
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
        # ANALYZED REVIEW DATA
        # ----------------------------------------------------

        st.subheader("📋 Analyzed Reviews")

        st.dataframe(
            dashboard_df,
            use_container_width=True,
            height=400
        )

        # ----------------------------------------------------
        # DOWNLOAD ANALYZED DATA
        # ----------------------------------------------------

        csv_data = dashboard_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Analyzed Reviews",
            data=csv_data,
            file_name="reviewiq_analyzed_reviews.csv",
            mime="text/csv"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ReviewIQ | Customer Review Analytics | "
    "TF-IDF + Logistic Regression"
)
