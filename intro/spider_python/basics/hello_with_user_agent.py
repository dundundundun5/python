import urllib.request
import http.client
# url为统一资源定位符
url = "http://httpbin.org/get"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"
}
# 创建请求对象，包装UA

req: urllib.request.Request = urllib.request.Request(url=url, headers=headers)
# 发送请求对象req，并获取相应对象
res: http.client.HTTPResponse= urllib.request.urlopen(req)
# res.read()返回bytes即字节码, 再decode()返回解码后的字符串
html = res.read().decode()
print(html)

