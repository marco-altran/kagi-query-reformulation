import requests
import time

# List of questions to test the /rewrite endpoint
questions = [
    "In what year was the winner of the 44th edition of the Miss World competition born?",
    "Who lived longer, Nikola Tesla or Milutin Milankovic?",
    "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?",
    "Create a table for top noise cancelling headphones that are not expensive",
    "What are some ways to do fast query reformulation",
    "Can you give me an easy pasta recipe?",
    "What are common flu symptoms?",
    "When was Taylor Swift born?",
    "Who won the 2022 World Cup?",
    "What does it mean a placenta grade 3?"
]

API_URL = "http://localhost:8080/rewrite"

def test_rewrite(question: str):
    """
    Sends a POST request to the /rewrite endpoint with the given question,
    measures elapsed time, and prints input, output, and timing.
    """
    payload = {"question": question, "desired_max_latency": 100}
    start = time.perf_counter()
    response = requests.post(API_URL, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code == 200:
        data = response.json()
        queries = data.get("queries", [])
    else:
        queries = []
        print(f"Error: received status code {response.status_code}")

    print(f"Input Question: {question}")
    print(f"Rewritten Queries: {queries}")
    print(f"Response Time: {elapsed_ms:.1f} ms")
    print("" + "-"*50 + "")

if __name__ == "__main__":
    for q in questions:
        test_rewrite(q)
