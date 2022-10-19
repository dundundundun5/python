# UA伪装
# User-Agent是请求载体的身份标识，如果为浏览器则判定为正常的请求
# 所以需要UA伪装
import requests
url = "https://www.baidu.com/s?"
# 处理url携带的参数 封装到字典中
# key_word = input("enter a word: ")
param = {
    'wd': "nihao"
}
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
}
# 对指定url发起请求时动态拼接参数
response = requests.get(url=url,params=param,headers=headers)
print(response.status_code)
print(response.text)