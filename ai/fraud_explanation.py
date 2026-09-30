import json

import ollama
import pandas as pd


def explain_fraud_prediction(
    amount,
    prediction,
    fraud_probability,
    model_version
):
    if prediction == 1 and fraud_probability >= 0.90:
        risk_level = "HIGH"
    elif prediction == 1 and fraud_probability >= 0.50:
        risk_level = "MEDIUM"
    elif prediction == 1:
        risk_level = "LOW"
    else:
        risk_level = "NORMAL"

    prompt = f"""
You are a fraud analytics assistant.

Analyze this machine learning prediction using ONLY the provided facts.

Transaction amount: {amount}
Model prediction: {prediction}
Fraud probability: {fraud_probability}
Risk level: {risk_level}
Model version: {model_version}

Return ONLY valid JSON with exactly these fields:

{{
  "risk_level": "...",
  "summary": "...",
  "recommendation": "..."
}}

Rules:
- The ML model makes the fraud prediction.
- Do not claim the transaction is definitely fraudulent.
- Do not describe fraud_probability as "100% confidence", "certainty", or model accuracy.
- Do not invent customer, merchant, location, behavioral, or transaction details.
- Do not explain anonymized V1-V28 features.
- Use the exact provided transaction amount.
- Keep the summary factual and concise.
- The recommendation should only say that the transaction should be reviewed.
- Do not mention a merchant because merchant information was not provided.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_response = response.message.content.strip()

    try:
        explanation = json.loads(raw_response)

        if not isinstance(explanation, dict):
            explanation = {}

    except json.JSONDecodeError:
        explanation = {}

    # Python controls deterministic fields.
    explanation["risk_level"] = risk_level

    # Keep the LLM output only as supporting text.
    summary = explanation.get("summary", "").strip()

    summary = summary.replace(
        "100% confidence",
        "fraud probability of 1.00"
    )
    summary = summary.replace(
        "100% probability",
        "fraud probability of 1.00"
    )

    if not summary or summary.startswith("{"):
        summary = (
            f"Transaction amount ${amount:.2f}. "
            f"The model predicted fraud and assigned a {risk_level} risk level."
        )

    explanation["summary"] = summary
    explanation["recommendation"] = "Review transaction"

    return explanation


if __name__ == "__main__":
    df = pd.read_csv("data/fraud_predictions.csv")

    df = (
    df[df["prediction"] == 1]
    .sort_values(
        by=["fraud_probability", "Amount"],
        ascending=[False, False]
    )
    .head(20)
)

    results = []

    for _, transaction in df.iterrows():

        explanation = explain_fraud_prediction(
            amount=float(transaction["Amount"]),
            prediction=int(transaction["prediction"]),
            fraud_probability=float(transaction["fraud_probability"]),
            model_version=str(transaction["model_version"])
        )

        results.append({
            "prediction_id": int(transaction["prediction_id"]),
            "amount": float(transaction["Amount"]),
            "prediction": int(transaction["prediction"]),
            "fraud_probability": float(transaction["fraud_probability"]),
            "risk_level": explanation["risk_level"],
            "summary": explanation["summary"],
            "recommendation": explanation["recommendation"],
            "model_version": str(transaction["model_version"])
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        "data/fraud_explanations.csv",
        index=False
    )

    print(f"\nSaved {len(output_df)} explanations.")
    print("Output: data/fraud_explanations.csv")