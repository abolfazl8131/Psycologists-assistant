import json
import requests


payload = {"inputs": ["what is mantal health?"]}

print(payload)
response = requests.post(
    url=f"http://localhost:1235/invocations",
    json=payload,
    
)


print(response.text)
   
