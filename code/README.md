cd ./backend
uvicorn app:app --reload


# api方式

import requests

url = "http://127.0.0.1:8000/api/predict"
data = {"text": "这个电影太烂了，浪费时间"}

response = requests.post(url, json=data)
print(response.json())
