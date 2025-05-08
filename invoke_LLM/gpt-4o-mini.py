import openai
import json
import chardet
import os
import pandas as pd
openai.api_key = ""
openai.base_url = ""


model = "gpt-4o-mini"
prompt=""""""
completion_LLM = openai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1000
    )
result=completion_LLM.choices[0].message.content
print(result)
