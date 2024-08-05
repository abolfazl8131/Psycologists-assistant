import json
import requests


payload = {"inputs": ["<HUMAN>:what is mantal health?"]}

print(payload)
response = requests.post(
    url=f"https://localhost:1234/invocations/",
    json=payload,
)


print(response.text)
   
