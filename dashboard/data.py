import pandas as pd


PREDICTIONS_PATH = "data/fraud_predictions.csv"


def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)

def get_all_predictions():
    return load_predictions()


def get_kpis():
    df = load_predictions()

    total_predictions = len(df)

    predicted_fraud = int(
        (df["prediction"] == 1).sum()
    )

    predicted_normal = int(
        (df["prediction"] == 0).sum()
    )

    predicted_fraud_rate = (
        predicted_fraud / total_predictions * 100
    )

    predicted_fraud_amount = df.loc[
        df["prediction"] == 1,
        "Amount"
    ].sum()

    average_fraud_amount = df.loc[
        df["prediction"] == 1,
        "Amount"
    ].mean()

    return {
        "total_predictions": total_predictions,
        "predicted_fraud": predicted_fraud,
        "predicted_normal": predicted_normal,
        "predicted_fraud_rate": predicted_fraud_rate,
        "predicted_fraud_amount": predicted_fraud_amount,
        "average_fraud_amount": average_fraud_amount
    }


def get_risk_data():
    df = load_predictions()

    df["risk_level"] = pd.cut(
        df["fraud_probability"],
        bins=[-0.01, 0.24, 0.49, 0.89, 1.0],
        labels=["NORMAL", "LOW", "MEDIUM", "HIGH"]
    )

    risk_data = (
        df["risk_level"]
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

    return risk_data


def get_performance_data():
    df = load_predictions()

    true_positive = int(
        ((df["Class"] == 1) & (df["prediction"] == 1)).sum()
    )

    false_positive = int(
        ((df["Class"] == 0) & (df["prediction"] == 1)).sum()
    )

    false_negative = int(
        ((df["Class"] == 1) & (df["prediction"] == 0)).sum()
    )

    true_negative = int(
        ((df["Class"] == 0) & (df["prediction"] == 0)).sum()
    )

    return pd.DataFrame({
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


def get_amount_data():
    df = load_predictions()

    fraud_df = df[df["prediction"] == 1]
    normal_df = df[df["prediction"] == 0]

    return pd.DataFrame({
        "Prediction": [
            "Predicted Fraud",
            "Predicted Normal"
        ],
        "Total Amount": [
            fraud_df["Amount"].sum(),
            normal_df["Amount"].sum()
        ]
    })


def get_fraud_transactions():
    predictions = load_predictions()

    explanations = pd.read_csv(
        "data/fraud_explanations.csv"
    )

    fraud_df = (
        predictions[predictions["prediction"] == 1]
        .sort_values(
            by=["fraud_probability", "Amount"],
            ascending=[False, False]
        )
        .head(20)
        .copy()
    )

    fraud_df["Risk Level"] = pd.cut(
        fraud_df["fraud_probability"],
        bins=[-0.01, 0.24, 0.49, 0.89, 1.0],
        labels=["NORMAL", "LOW", "MEDIUM", "HIGH"]
    )

    fraud_df = fraud_df.merge(
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

    fraud_df["risk_level"] = fraud_df["risk_level"].fillna(
        fraud_df["Risk Level"].astype(str)
    )

    fraud_df["summary"] = fraud_df["summary"].fillna(
        "AI explanation not generated for this transaction"
    )

    fraud_df["recommendation"] = fraud_df["recommendation"].fillna(
        "Review transaction"
    )

    return fraud_df[
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
    ].rename(columns={
        "prediction_id": "Transaction ID",
        "Amount": "Amount",
        "Class": "Actual Class",
        "prediction": "Prediction",
        "fraud_probability": "Fraud Probability",
        "risk_level": "Risk Level",
        "summary": "AI Explanation",
        "recommendation": "Recommendation",
        "model_version": "Model Version"
    })