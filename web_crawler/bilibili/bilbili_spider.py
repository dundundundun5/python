# 已知视频的url，用户的cookie，爬取视频的cid和aid

def get_cid_n_aid(bv):
    import re, requests
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }
    url = f"https://www.bilibili.com/video/{bv}"
    r = requests.get(url=url, headers=headers)
    res = []
    for ch in ['a', 'c']:
        regex = re.compile(f'\"{ch}id\":[0-9]*')
        pos = re.search(regex, r.text)
        raw = r.text[pos.start():pos.end()]
        id = raw.split(":")[1]
        res.append(id)
    return res


def get_comments_n_save(bv):
    file = f'D:/temp_files/{bv}_comments.csv'
    aid, cid = get_cid_n_aid(bv)
    import requests,re,time
    url = "https://comment.bilibili.com/" + cid + ".xml"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    }
    r = requests.get(url=url, headers=headers)

    lines = r.content.decode('utf-8')
    # 正则表达式筛选掉前面无关内容
    pattern = "\</source\>"
    temp = re.search(pattern=pattern, string=lines)
    pos = temp.end()
    # 找到无关内容的位置并取子串
    lines = lines[pos:]
    # 提取评论
    comment_message: list[str] = re.findall(pattern=">(.*?)</d>", string=lines, flags=re.I)
    # 提取弹幕参数
    params: list[list[str]] = re.findall(pattern="\"\S+\"", string=lines, flags=re.I)
    # 用于存放转换前的数据
    raw_uids = []
    raw_times = []
    # 用于存放转换后的数据
    user_id_crc = []
    comment_datetime = []
    # 用逗号分割弹幕参数，并获取原始数据
    for param in params:
        a = re.split(',', string=param, maxsplit=0, flags=re.I)
        raw_uids.append(a[6])
        raw_times.append(a[4])
    # unix时间戳->本地时间
    for raw_time in raw_times:
        local_time = time.localtime(int(raw_time))
        res = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        comment_datetime.append(res)
    # uid和校验码的换算要另外进行
    user_id_crc = raw_uids
    # 换一种格式写入源文件
    import pandas as pd
    res =  pd.DataFrame(data=None, columns=['comment_datetime', 'user_id_crc', 'comment_message'])
    for i in range(len(comment_datetime)):
        res.loc[len(res)] = [comment_datetime[i],user_id_crc[i], comment_message[i]]
    res.to_csv(file, index=False)


import binascii
# 计算uid（ascii）的crc循环校验码并转为十六进制

def crc32ascii(v):
    v = v.encode('utf-8')
    return '0x%8x' % (binascii.crc32(v) & 0xffffffff)


def get_root_replies_n_save(bv):
    file = f'D:/temp_files/{bv}_replies.csv'
    import requests
    oid, cid = get_cid_n_aid(bv)# oid就是aid
    i = 1 # 评论页码，一页含20层
    import pandas as pd
    res = pd.DataFrame(data=None,
                       columns=['reply_id', 'user_id', 'user_id_crc',
                                'user_name', 'reply_datetime', 'reply_message', 'like'])
    while True:
        url = f'https://api.bilibili.com/x/v2/reply/main?jsonp=jsonp&next={i}&type=1&oid={oid}&mode=3&plat=1'
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
        'cookie': "_uuid=10E1C610F10-D282-A7CB-153A-77CDEB1A2C4F00293infoc; buvid3=4E5D4D4E-DAAD-0A70-1821-85AD8FED9F0203174infoc; b_nut=1645259602; buvid_fp_plain=undefined; CURRENT_BLACKGAP=0; blackside_state=0; rpdid=|(u~||ku)mu)0J'uYRlRmlmmu; nostalgia_conf=-1; hit-dyn-v2=1; LIVE_BUVID=AUTO8916501095459182; fingerprint3=767d27043ccdfc93aa101296c76e1a1f; i-wanna-go-back=-1; DedeUserID=15486269; DedeUserID__ckMd5=cc54d3c7f6415ba4; b_ut=5; buvid4=77C9F841-CA24-A1B4-CB17-D0F3CED14F1303174-022021916-5mepP/ytJSskvaMc8sbZKg==; CURRENT_QUALITY=0; fingerprint=ce40c5fd195b7eb72e8c7e477534ef48; buvid_fp=ce40c5fd195b7eb72e8c7e477534ef48; CURRENT_FNVAL=4048; hit-new-style-dyn=0; SESSDATA=e90b5863,1683373548,0dbcb*b2; bili_jct=8ef2d66658554603424261c771047842; b_lsid=ADA24410A_184552F662F; bsource=search_baidu; go_old_video=1; bp_video_offset_15486269=726020421484281900; innersign=1; PVID=2; sid=hgkh62jg"
        }
        response = requests.get(url=url, headers=headers)
        raw_replies = response.json() # 获取dict格式的json响应
        if raw_replies['data']['cursor']['is_end']:
            break # 评论翻到底了
        raw_replies = raw_replies['data']['replies'] # 进入dict的评论数据区
        for raw_reply in raw_replies: # 根据json分析出的规律，仅适用于b站视频评论
            reply_id = raw_reply['rpid']
            user_id = raw_reply['member']['mid'] # 写作mid，读作uid
            user_id_crc = crc32ascii(user_id)
            user_id_crc = str(user_id_crc)[2:]
            user_name = raw_reply['member']['uname']
            import time
            r_datetime = raw_reply['ctime']
            local_time = time.localtime(int(r_datetime))
            reply_datetime = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
            reply_message = raw_reply['content']['message']
            like = raw_reply['like']
            data = [reply_id, user_id, user_id_crc, user_name, reply_datetime, reply_message, like]
            res.loc[len(res)] = data
        i = i + 1
    res.to_csv(file, index=False)

if __name__ == '__main__':
    bv = 'BV1rd4y1C7o6'
    get_comments_n_save(bv)
    get_root_replies_n_save(bv)
    import pandas as pd
    file_c = f'D:/temp_files/{bv}_comments.csv'
    file_r = f'D:/temp_files/{bv}_replies.csv'
    comments = pd.read_csv(file_c)
    replies = pd.read_csv(file_r)
    input()