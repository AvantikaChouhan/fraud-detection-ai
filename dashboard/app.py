import streamlit as st
import pandas as pd
import plotly.express as px

from data import get_all_predictions


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="FraudStream AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .dashboard-header {
        padding: 1rem 0 0.5rem 0;
    }

    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        opacity: 0.7;
    }

    .status-card {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        font-weight: 600;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        opacity: 0.65;
        margin-bottom: 0.8rem;
    }

    .alert-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(220, 53, 69, 0.35);
        background-color: rgba(220, 53, 69, 0.08);
    }

    .ai-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

predictions_df = get_all_predictions().copy()

predictions_df["Risk Level"] = pd.cut(
    predictions_df["fraud_probability"],
    bins=[-0.01, 0.24, 0.49, 0.89, 1.0],
    labels=["NORMAL", "LOW", "MEDIUM", "HIGH"]
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 🔍 FraudStream AI")

    st.caption("Fraud Detection & AI Analytics")

    st.divider()

    st.markdown("### 🎛️ Dashboard Controls")

    risk_options = [
        "ALL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "NORMAL"
    ]

    selected_risk = st.selectbox(
        "Risk Level",
        risk_options
    )

    probability_range = st.sidebar.slider(
    "Fraud Probability",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01
)

    search_transaction = st.text_input(
        "🔎 Search Transaction ID",
        placeholder="e.g. 2184"
    )

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("**Model:** Logistic Regression")
    st.write("**Version:** logistic_regression_v1")
    st.write("**LLM:** Llama 3.2 3B")
    st.write("**GenAI:** Ollama")

    st.divider()

    st.caption(
        "FraudStream AI — Data Engineering + ML + GenAI"
    )


# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

filtered_predictions = predictions_df.copy()


if selected_risk != "ALL":

    filtered_predictions = filtered_predictions[
        filtered_predictions["Risk Level"] == selected_risk
    ]


filtered_predictions = filtered_predictions[
    (filtered_predictions["fraud_probability"] >= probability_range[0])
    &
    (filtered_predictions["fraud_probability"] <= probability_range[1])
]


if search_transaction:

    filtered_predictions = filtered_predictions[
        filtered_predictions["prediction_id"]
        .astype(str)
        .str.contains(
            search_transaction,
            case=False,
            na=False
        )
    ]


# ---------------------------------------------------------
# DYNAMIC KPIs
# ---------------------------------------------------------

total_predictions = len(filtered_predictions)

predicted_fraud = int(
    (filtered_predictions["prediction"] == 1).sum()
)

predicted_normal = int(
    (filtered_predictions["prediction"] == 0).sum()
)


if total_predictions > 0:

    predicted_fraud_rate = (
        predicted_fraud / total_predictions * 100
    )

else:

    predicted_fraud_rate = 0


predicted_fraud_amount = filtered_predictions.loc[
    filtered_predictions["prediction"] == 1,
    "Amount"
].sum()


average_fraud_amount = filtered_predictions.loc[
    filtered_predictions["prediction"] == 1,
    "Amount"
].mean()


if pd.isna(average_fraud_amount):

    average_fraud_amount = 0


# ---------------------------------------------------------
# DYNAMIC RISK DATA
# ---------------------------------------------------------

risk_data = (
    filtered_predictions["Risk Level"]
    .value_counts()
    .reindex(
        ["HIGH", "MEDIUM", "LOW", "NORMAL"],
        fill_value=0
    )
    .reset_index()
)

risk_data.columns = [
    "Risk Level",
    "Transactions"
]


# ---------------------------------------------------------
# DYNAMIC MODEL PERFORMANCE
# ---------------------------------------------------------

true_positive = int(
    (
        (filtered_predictions["Class"] == 1)
        &
        (filtered_predictions["prediction"] == 1)
    ).sum()
)

false_positive = int(
    (
        (filtered_predictions["Class"] == 0)
        &
        (filtered_predictions["prediction"] == 1)
    ).sum()
)

false_negative = int(
    (
        (filtered_predictions["Class"] == 1)
        &
        (filtered_predictions["prediction"] == 0)
    ).sum()
)

true_negative = int(
    (
        (filtered_predictions["Class"] == 0)
        &
        (filtered_predictions["prediction"] == 0)
    ).sum()
)


performance_data = pd.DataFrame({
    "Result": [
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative"
    ],
    "Transactions": [
        true_positive,
        false_positive,
        false_negative,
        true_negative
    ]
})


# ---------------------------------------------------------
# DYNAMIC AMOUNT DATA
# ---------------------------------------------------------

fraud_amount = filtered_predictions.loc[
    filtered_predictions["prediction"] == 1,
    "Amount"
].sum()

normal_amount = filtered_predictions.loc[
    filtered_predictions["prediction"] == 0,
    "Amount"
].sum()


amount_data = pd.DataFrame({
    "Prediction": [
        "Predicted Fraud",
        "Predicted Normal"
    ],
    "Total Amount": [
        fraud_amount,
        normal_amount
    ]
})


# ---------------------------------------------------------
# FRAUD INVESTIGATION DATA
# ---------------------------------------------------------

fraud_transactions = (
    filtered_predictions[
        filtered_predictions["prediction"] == 1
    ]
    .sort_values(
        by=["fraud_probability", "Amount"],
        ascending=[False, False]
    )
    .head(20)
    .copy()
)


explanations = pd.read_csv(
    "data/fraud_explanations.csv"
)


fraud_transactions = fraud_transactions.merge(
    explanations[
        [
            "prediction_id",
            "risk_level",
            "summary",
            "recommendation"
        ]
    ],
    on="prediction_id",
    how="left"
)


fraud_transactions["risk_level"] = (
    fraud_transactions["risk_level"]
    .fillna(
        fraud_transactions["Risk Level"].astype(str)
    )
)


fraud_transactions["summary"] = (
    fraud_transactions["summary"]
    .fillna(
        "AI explanation not generated for this transaction"
    )
)


fraud_transactions["recommendation"] = (
    fraud_transactions["recommendation"]
    .fillna("Review transaction")
)


fraud_transactions = fraud_transactions[
    [
        "prediction_id",
        "Amount",
        "Class",
        "prediction",
        "fraud_probability",
        "risk_level",
        "summary",
        "recommendation",
        "model_version"
    ]
].rename(
    columns={
        "prediction_id": "Transaction ID",
        "Amount": "Amount",
        "Class": "Actual Class",
        "prediction": "Prediction",
        "fraud_probability": "Fraud Probability",
        "risk_level": "Risk Level",
        "summary": "AI Explanation",
        "recommendation": "Recommendation",
        "model_version": "Model Version"
    }
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="dashboard-header">

    <div class="dashboard-title">
    🔍 FraudStream AI
    </div>

    <div class="dashboard-subtitle">
    Fraud Detection & AI Analytics Platform
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


header_col1, header_col2 = st.columns([5, 1])

with header_col2:

    st.markdown(
        """
        <div class="status-card">
        🟢 Model Online
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ---------------------------------------------------------
# ACTIVE FILTER STATUS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🎛️ Active Dashboard Filters</div>',
    unsafe_allow_html=True
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:

    st.metric(
        "Risk Filter",
        selected_risk
    )

with filter_col2:

    st.metric(
        "Probability Range",
        f"{probability_range[0]:.2f} – {probability_range[1]:.2f}"
    )

with filter_col3:

    st.metric(
        "Matching Transactions",
        f"{len(filtered_predictions):,}"
    )


# ---------------------------------------------------------
# KPI SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Fraud Detection Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Metrics dynamically calculated from the currently filtered dataset.'
    '</div>',
    unsafe_allow_html=True
)


kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:

    st.metric(
        "Total Predictions",
        f"{total_predictions:,}"
    )

with kpi2:

    st.metric(
        "🚨 Predicted Fraud",
        f"{predicted_fraud:,}"
    )

with kpi3:

    st.metric(
        "Predicted Normal",
        f"{predicted_normal:,}"
    )


kpi4, kpi5, kpi6 = st.columns(3)

with kpi4:

    st.metric(
        "Fraud Rate",
        f"{predicted_fraud_rate:.2f}%"
    )

with kpi5:

    st.metric(
        "Fraud Amount",
        f"${predicted_fraud_amount:,.2f}"
    )

with kpi6:

    st.metric(
        "Avg. Fraud Amount",
        f"${average_fraud_amount:,.2f}"
    )


# ---------------------------------------------------------
# ALERT
# ---------------------------------------------------------

st.markdown("")

if predicted_fraud > 0:

    st.markdown(
        f"""
        <div class="alert-box">

        🚨 <b>{predicted_fraud:,} transactions</b> are currently
        flagged by the fraud detection model.

        Fraud prediction rate:
        <b>{predicted_fraud_rate:.2f}%</b>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.info(
        "No fraud predictions match the current filters."
    )


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Risk & Model Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Charts automatically update when dashboard filters change.'
    '</div>',
    unsafe_allow_html=True
)


chart_col1, chart_col2 = st.columns(2)


# ---------------------------------------------------------
# RISK CHART
# ---------------------------------------------------------

with chart_col1:

    fig_risk = px.bar(
        risk_data,
        x="Risk Level",
        y="Transactions",
        text="Transactions",
        title="Fraud Risk Distribution"
    )

    fig_risk.update_traces(
        textposition="outside"
    )

    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Transactions",
        height=420,
        showlegend=False
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )


# ---------------------------------------------------------
# PERFORMANCE CHART
# ---------------------------------------------------------

with chart_col2:

    fig_performance = px.bar(
        performance_data,
        x="Result",
        y="Transactions",
        text="Transactions",
        title="Model Prediction Results"
    )

    fig_performance.update_traces(
        textposition="outside"
    )

    fig_performance.update_layout(
        xaxis_title="Prediction Result",
        yaxis_title="Transactions",
        height=420,
        showlegend=False
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )


# ---------------------------------------------------------
# AMOUNT ANALYSIS
# ---------------------------------------------------------

fig_amount = px.bar(
    amount_data,
    x="Prediction",
    y="Total Amount",
    text="Total Amount",
    title="Transaction Amount by Prediction"
)

fig_amount.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside"
)

fig_amount.update_layout(
    xaxis_title="Prediction",
    yaxis_title="Total Amount",
    height=420,
    showlegend=False
)

st.plotly_chart(
    fig_amount,
    use_container_width=True
)


# ---------------------------------------------------------
# FRAUD INVESTIGATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🚨 Fraud Investigation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Investigate the highest-risk transactions within the current filter.'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    f"Showing **{len(fraud_transactions)}** top predicted-fraud transactions."
)

if len(fraud_transactions) > 0:

    display_df = fraud_transactions.copy()

    display_df["Fraud Probability"] = (
        display_df["Fraud Probability"] * 100
    ).round(2).astype(str) + "%"

    display_df["Amount"] = display_df[
        "Amount"
    ].map(
        lambda x: f"${x:,.2f}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=520
    )

else:

    st.info(
        "No predicted-fraud transactions match the current filters."
    )


# ---------------------------------------------------------
# AI INVESTIGATION ASSISTANT
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🤖 AI Investigation Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Select a transaction to inspect its AI-generated explanation.'
    '</div>',
    unsafe_allow_html=True
)

if len(fraud_transactions) > 0:

    selected_transaction_id = st.selectbox(
        "Select Transaction",
        fraud_transactions["Transaction ID"].tolist()
    )

    selected_row = fraud_transactions[
        fraud_transactions["Transaction ID"]
        == selected_transaction_id
    ].iloc[0]

    ai_col1, ai_col2, ai_col3 = st.columns(3)

    with ai_col1:
        st.metric(
            "Transaction Amount",
            f"${selected_row['Amount']:,.2f}"
        )

    with ai_col2:
        st.metric(
            "Fraud Probability",
            f"{selected_row['Fraud Probability']:.2%}"
        )

    with ai_col3:
        st.metric(
            "Risk Level",
            selected_row["Risk Level"]
        )

    st.markdown(
        f"""
        <div class="ai-box">

        <b>🤖 AI Explanation</b>

        <br><br>

        {selected_row["AI Explanation"]}

        <br><br>

        <b>Recommendation:</b>
        {selected_row["Recommendation"]}

        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)


with footer_col1:

    st.caption(
        "FraudStream AI | Data Engineering + Machine Learning + GenAI"
    )


with footer_col2:

    st.caption(
        "Model: logistic_regression_v1 | LLM: Llama 3.2 3B"
    )