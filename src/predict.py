# src/predict.py
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

# Test samples (one example from each category)
test_tickets = [
    {
        "Subject": "App crashes on login",
        "Description": "The app closes every time I try to log in"
    },
    {
        "Subject": "Add dark mode option",
        "Description": "Would love to see a dark theme feature in the next update"
    },
    {
        "Subject": "VPN blocking service",
        "Description": "Our office VPN blocks your application. Need help configuring"
    },
    {
        "Subject": "Payment not processed",
        "Description": "My credit card was charged but the subscription is still inactive"
    },
    {
        "Subject": "Unable to reset password",
        "Description": "The password reset link is not working for my account"
    }
]

def predict_ticket(payload):
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}

if __name__ == "__main__":
    print("🚀 Testing multiple tickets...\n")
    for idx, ticket in enumerate(test_tickets, start=1):
        result = predict_ticket(ticket)
        print(f"--- Test Case {idx} ---")
        print("Input:", json.dumps(ticket, indent=4))
        print("Output:", json.dumps(result, indent=4))
        print("\n")
