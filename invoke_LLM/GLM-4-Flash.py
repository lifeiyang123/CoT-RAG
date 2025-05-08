from zhipuai import ZhipuAI
import json
import chardet
import os
import pandas as pd

client = ZhipuAI(api_key="") 

prompt=""" """


response = client.chat.completions.create(
                    model="GLM-4-Flash",  
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
print(response.choices[0].message.content)

