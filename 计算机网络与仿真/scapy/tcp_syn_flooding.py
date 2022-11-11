from scapy.all import *
from scapy.layers.inet import *


def tcp_syn(target_ip, sport, dport):
    src = RandIP()
    dst = target_ip

    pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, seq=1505066, flags='S')
    send(pkt)


if __name__ == '__main__':
    pass
    # TODO
