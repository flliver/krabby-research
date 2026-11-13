"""Physical USB-backpressure/CAN/watchdog regression; never energizes drivers."""
import json,time
from pathlib import Path
from krabby import *
from openblt_canable import ROOT

def main():
    n=Node(); usb=None
    try:
        for m in (1,2,3):n.set(m,10,1)
        n.set(0,1,5);n.request(ARM);n.motor(0,0,0)
        before=n.status()
        # 400 diagnostic frames while the STM32 USB CDC has no reader. These
        # must not block application dispatch or the independent watchdog.
        n.link.port.write(b't35080102030405060708\r'*400)
        time.sleep(.2)
        state=n.status()
        assert not state['node_status']&8 and state['node_status']&4 and state['commands']==[0,0,0],state
        assert state['samples']>before['samples'] and state['adc_errors']==0,state
        print('PASS watchdog and acquisition survive USB diagnostic backpressure',state,flush=True)
        delays=[]
        for i in range(500):
            t=time.perf_counter();state=n.status();delays.append(time.perf_counter()-t)
            assert state['adc_errors']==0 and not state['node_status']&8
        usb=Node('COM3',usb=True)
        assert usb.request(VERSION)==n.request(VERSION)
        for i in range(100):
            assert usb.page(i%4)==n.page(i%4)
        # One real board, five absent-node addresses: representative request
        # traffic only, not a claim of six physical nodes or their replies.
        n.set(0,1,25);n.request(ARM);stream_start=time.monotonic()
        for cycle in range(50):
            for absent in range(1,6):
                n.link.send(bytes((1,cycle,11,0,0,0,0,0,0)),0x500+absent)
            n.motor(0,0,0)
            time.sleep(max(0,stream_start+(cycle+1)*.02-time.monotonic()))
        assert n.status()['node_status']&8
        n.request(DISARM)
        # Human USB commands dispatch the identical backend.
        for command in (b'node version\n',b'node page 1\n',b'node sensors 0\n'):
            usb.link.write(command);deadline=time.monotonic()+1
            while time.monotonic()<deadline:
                line=usb.link.readline().strip()
                if line.startswith(b'APP '):
                    response=bytes.fromhex(line[4:].decode());assert response[3]==0;break
            else:raise TimeoutError('Human USB command response missing')
        usb.link.write(b'can status\n');deadline=time.monotonic()+1;bridge=''
        while time.monotonic()<deadline:
            line=usb.link.readline().decode(errors='replace').strip()
            if line.startswith('irq='):bridge=line;break
        assert bridge
        state=n.status()
        result=dict(result='PASS',diagnostic_frames=400,can_status_queries=500,cross_transport_page_comparisons=100,max_query_ms=max(delays)*1000,bridge_diagnostics=bridge,human_usb_commands=3,emulated_request_addresses=6,physical_nodes=1,motor_cycles_50Hz=50,final_status=state)
        (ROOT/'bench/application/stress_results.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS USB recovery and CAN/USB consistency',result)
    finally:
        n.close()
        if usb:usb.close()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware',action='store_true',required=True,help='Explicitly allow fixture configuration changes and watchdog tests')
    parser.parse_args()
    main()
