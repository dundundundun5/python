# scapy速写记录
* 使用

    1. IP(ttl=,src=,dst=)
    2. lsc()
    3. TCP(sport=,dport=)
    4. IP()/TCP()
    5. send(IP()/TCP(sport=[RandShort()]*10,dport=,flags="S"), count=)
    6. sniff(iface=,count=, prn=lambda x : )
    7. a = _, a.summary(), a.show()
    8. ctrl+C STOP
    9. p = rdpcap(), pkt=p[INDEX]
    10. ls(pkt),hexdump(pkt),dir(pkt)
    11. send(pkt/ICMP()/"RAW_DATA",return_packets=True)
    12. 127.0.0.1
    13. sendp(Ether()/"WORDS",iface=,return_packets=)
    14. sr1(IP()/ICMP()/"dun")
    15. srloop()
    16.  p = sr1(IP(dst="8.8.8.8")/UDP()/DNS(rd=1,qd=DNSQR(qname="www.baidu.com")))
    17.  sr() ,ans,unans= _
    18.  for snd,rsv in ans:
         1.  print(rsv.show())
    19. plot()

# 概念
* arp spoofing
* XSS跨站脚本攻击