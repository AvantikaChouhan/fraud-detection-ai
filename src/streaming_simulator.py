import pandas as pd
import json
import socket
import time


CSV_PATH = "data/creditcard.csv"
HOST = "localhost"
PORT = 9999
DELAY_SECONDS = 1


def stream_transactions():
    df = pd.read_csv(CSV_PATH)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Waiting for Spark connection on {HOST}:{PORT}...")

    conn, address = server.accept()

    print(f"Spark connected from {address}")
    print(f"Total transactions: {len(df)}")
    print("Starting transaction stream...\n")

    for index, row in df.iterrows():

        transaction = {
            "transaction_id": index + 1,
            "time": float(row["Time"]),
            "amount": float(row["Amount"]),
            "class": int(row["Class"]),
        }

        message = json.dumps(transaction) + "\n"

        conn.sendall(message.encode("utf-8"))

        print(f"Sent: {transaction}")

        time.sleep(DELAY_SECONDS)

    conn.close()
    server.close()


if __name__ == "__main__":
    stream_transactions()