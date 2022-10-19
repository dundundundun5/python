import requests
import json
url = "https://fanyi.baidu.com/sug"
word = input()
# 参数处理
data={
    "kw": word
}
# UA伪装
headers={
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
}
# 请求发送
response = requests.post(url=url,data=data,headers=headers,)
# 获取响应数据json()返回的是字典对象，仅在响应数据是json才行
# 查看抓包工具中响应头的Content-Type
dict_obj = response.json()
fp = open(file=word+".json",mode="w",encoding="utf-8")
json.dump(obj=dict_obj,fp=fp,ensure_ascii=False)
fp.close()