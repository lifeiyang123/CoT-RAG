import json
import chardet
import os
import qianfan
import pandas as pd
 
data_list = []
file_name = ""  
with open(file_name, 'r',encoding=encoding) as file:  
    lines=file.readlines()
    for line in lines:
        line=line.rstrip('\n')
        data_list.append(json.loads(line))  
#print(data_list[0]["question"])



os.environ["QIANFAN_ACCESS_KEY"] = ""
os.environ["QIANFAN_SECRET_KEY"] = ""
chat_comp = qianfan.ChatCompletion()

with open('','w',encoding="utf-8") as f1:
    
    for i in range(0,100):

        prompt="'Question':"+data_list[i]["question"]+','+"'Options:'"+str(data_list[i]["options"])
        resp = chat_comp.do(model="ERNIE-3.5-128K", messages=[{
            "role": "user",
            "content":prompt
        }])
        print(resp["body"]["result"])
        f1.write(str(i)+resp["body"]["result"]+'\n')
       


   