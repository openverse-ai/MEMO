import requests
import json
import os
import dotenv
import time
from datetime import datetime

dotenv.load_dotenv()

while True:
    response = requests.get(
        url="https://openrouter.ai/api/v1/key",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
        }
    )
    
    data = response.json()
    print(data)
    usage = data["data"]["usage"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{timestamp} - Usage: {usage}")
    
    time.sleep(30)