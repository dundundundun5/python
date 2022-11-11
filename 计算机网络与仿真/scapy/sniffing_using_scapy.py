from scapy.all import sniff
print(help(sniff))
# timeout=在一定时间后停止嗅探
# pkts_list = sniff(timeout=5)
from scapy.all import wrpcap
print(help(wrpcap))
#  将嗅探到的所有包，存储为 NAME.pcap
# wrpcap('scapy.pcap', pkts_list)
quit()