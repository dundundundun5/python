# 已知视频cid，爬取视频的弹幕、用户标识、发弹幕时间
import requests
import re
import time


def download_n_save(cid: str, file:str):
    url = "https://comment.bilibili.com/" + cid + ".xml"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    }
    r = requests.get(url=url, headers=headers)
    # 以字节流写入
    f = open(file=file, mode="wb")
    # 获取响应报文的二进制表示
    f.write(r.content)
    # 关闭文件对象
    f.close()


def trans_n_save(file):
    f = open(file=file, mode="r", encoding="utf-8")
     # 一次读取全部行
    lines:list[str] = f.readlines()
    f.close()
    # 由于数据全是一行的，所以列表就一个元素
    lines:str = lines[0]
    # 正则表达式筛选掉前面无关内容
    pattern = "\</source\>"
    temp = re.search(pattern=pattern, string=lines)
    pos = temp.end()
    # 找到无关内容的位置并取子串
    lines = lines[pos:]
    # 提取评论
    comments:list[str] = re.findall(pattern=">(.*?)</d>", string=lines, flags=re.I)
    # 提取弹幕参数
    params:list[list[str]] = re.findall(pattern="\"\S+\"", string=lines, flags=re.I)
    # 用于存放转换前的数据
    raw_uids = []
    raw_times = []
    # 用于存放转换后的数据
    uids = []
    times = []
    # 用逗号分割弹幕参数，并获取原始数据
    for param in params:
        a = re.split(pattern=",", string=param, maxsplit=0, flags=re.I)
        raw_uids.append(a[6])
        raw_times.append(a[4])
    # unix时间戳->本地时间
    for raw_time in raw_times:
        local_time = time.localtime(int(raw_time))
        res = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        times.append(res)
    # uid和校验码的换算要另外进行
    uids = raw_uids
    # 换一种格式写入源文件
    f = open(file=file, mode="w", encoding="utf-8")
    for i in range(len(times)):
        f.write(f"{times[i]},{uids[i]},{comments[i]}\n")
    # 关闭文件对象
    f.close()

# 分p视频各有各的cid
cid = "882072634"
file = cid + ".txt"
download_n_save(cid, file)
trans_n_save(file)


