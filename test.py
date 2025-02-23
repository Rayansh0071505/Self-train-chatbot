# demo_test_chat.py
import requests

BACKEND_URL = "http://localhost:8080"  # or wherever your FastAPI is running
USER_ID = "3f7825b3-62a9-4a9c-a20a-50373cd2be35"       # replace with a valid user_id from Mongo

def send_chat(message):
    url = f"{BACKEND_URL}/chat/{USER_ID}"
    payload = {
        "query": message,
        "user_id": USER_ID
    }
    r = requests.post(url, json=payload)
    print("User says:", message)
    if r.status_code == 200:
        print("Backend response:", r.json())
    else:
        print("Error:", r.status_code, r.text)
    print("-" * 60)

def run_demo():
    # Example conversation flow:
    send_chat("hi")
    send_chat("i want to buy adidas shoes")
    send_chat("show more")
    send_chat("Vendor: adidas")  # currently triggers the repeated vendor list
    send_chat("thank you")

if __name__ == "__main__":
    run_demo()
