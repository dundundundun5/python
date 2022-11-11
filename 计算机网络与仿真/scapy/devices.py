
# 此电脑的window10无线网卡，没有监听模式，所以无法监听802.11帧，终结
from scapy.all import *
res = []
def prn(pkt):
    if pkt.haslayer(Dot11):
        a = pkt.getlayer(Dot11)
        res.append(a.addr2)
        print(a.addr2)
sniff(iface=conf.iface, count=1000, prn=prn)
input()