from scapy.all import *
from scapy.layers.dns import *
from scapy.layers.inet import *
# <https://www.youtube.com/watch?v=n3yyV96LTTI>
# scapy.conf配置文件
# print(conf)
# 网络接口
# print(conf.iface)
# creating DNS pkt with Scapy
# rd=1 requesting recursion qd=DNSQR(URL) DNS QUERY
# dns_pkt1 = IP(src='192.168.1.139', dst='192.168.1.1') / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname='www.baidu.com'))
# print(dns_pkt1.show())
# sending dns_pkt1 for 5 times and getting the return packets
# sent = send(dns_pkt1, count=5, return_packets=True)
# print(sent.summary())
# ========================================
# 真实场景，直接请求域名服务器
# dns_pkt2 = IP(dst='114.114.114.114') / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname='www.baidu.com'))
# print(dns_pkt2.show())
# response = srloop(dns_pkt2, count=5)
# ===================================
# 基于https的dns,DOH,概念比较复杂，了解即可
import base64
dns_msg = DNS(rd=1, qd=DNSQR(qname='www.baidu.com'))
dns_msg_str = base64.standard_b64encode(raw(dns_msg)).decode('utf8')
print(type(dns_msg_str), dns_msg_str)
url = "https://9.9.9.9:443/dns-query?dns=" + dns_msg_str
print("curl -i",url)
# win10 curl没配置，无法验证dns-query
