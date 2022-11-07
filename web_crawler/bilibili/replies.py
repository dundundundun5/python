import requests
import time
# 参考来源
# https://www.cnblogs.com/Kled/p/16023225.html

class JsonProcess:
    def __init__(self):
        self.Json_data = ''
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }

    def spider(self, URL):
        url = URL
        response = requests.get(url, headers=self.headers,verify=False)
        response.encoding = 'utf-8'
        self.Json_data = response.json()['data']['replies']

# 爬取子评论
def getSecondReplies(root):
    reply = JsonProcess()
    # page
    pn = 1
    # hard loops
    while True:
        url = f'https://api.bilibili.com/x/v2/reply/reply?jsonp=jsonp&pn={pn}&type=1&oid=979849123&ps=10&root={root}&_=1647581648753'
        time.sleep(1)
        reply.spider(url)
        if reply.Json_data is None:
            break
        for node in reply.Json_data:
            rpid = node['rpid']
            name = node['member']['uname']
            avatar = node['member']['avatar']
            content = node['content']['message']
            data = (rpid, root, name, avatar, content)
            print(data)
            # TODO 存储评论
        pn = pn + 1

# 爬取根评论
def getReplies(jp:JsonProcess, i):
    while True:
        url = f'https://api.bilibili.com/x/v2/reply/main?jsonp=jsonp&next={i}&type=1&oid=979849123&mode=3&plat=1&_=1647577851745'
        jp.spider(url)
        if jp.Json_data is None:
            break
        for node in jp.Json_data:
            rpid = node['rpid']
            name = node['member']['uname']
            avatar = node['member']['avatar']
            content = node['content']['message']
            data = (rpid, '0', name, avatar, content)
            print(data)

