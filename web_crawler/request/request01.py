
import requests
# 1 set url
url = "https://www.bilibili.com/robots.txt"
# 2 request
response = requests.get(url=url)
# 3 fetch the answer 返回字符串形式的响应数据
page = response.text
print(page)
# 4 持久化存储
with open(file="./request01.txt", mode="w",encoding="utf-8") as f:
    f.write(page)
    f.close()