import requests
resp = requests.get(
    "https://www.nawaloka.com",
    headers={"User-Agent": "Mozilla/5.0 (compatible; HospitalChatbotBot/1.0)"},
    timeout=10,
)
print("Status code:", resp.status_code)
print("Content-Type:", resp.headers.get("Content-Type"))
print("Raw HTML length:", len(resp.text))
print("First 500 chars:\n", resp.text[:500])