import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI(
  api_key=os.getenv('AI_GATEWAY_API_KEY'),
  base_url='https://ai-gateway.vercel.sh/v1'
)
 
response = client.chat.completions.create(
  model='xai/grok-4',
  messages=[
    {
      'role': 'user',
      'content': 'Why is the sky blue?'
    }
  ]
)
print(response.choices[0].message.content)