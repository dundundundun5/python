import http.client
import urllib.request
import os
# 发起一个请求
response: http.client.HTTPResponse = urllib.request.urlopen("http://www.baidu.com/")
# 输出html信息
bytes = response.read() # type: bytes
# decode()方法返回字符串
string = response.read().decode() # type: str
# 获取url
url: str = response.geturl()
# 获取状态码
code = response.getcode()
# 通过向 HTTP 测试网站（http://httpbin.org/）发送 GET 请求来查看请求头信息，从而获取爬虫程序的 UA
response = urllib.request.urlopen("http://httpbin.org/get")
html = response.read().decode()
# 首部的"User-Agent": "Python-urllib/3.10"，很明显是把爬虫写在了自己的脸上
print(html)
os.system("pause")