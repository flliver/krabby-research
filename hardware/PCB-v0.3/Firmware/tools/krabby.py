"""Krabby protocol v1 reference host. CANable serial or independent USB service."""
import argparse
import struct
import time
import serial
from openblt_canable import Canable

VERSION,STATUS,SENSORS,GET,SET,DEFAULTS,SAVE,LOAD,ARM,DISARM,MOTOR,REBOOT_ARM,REBOOT_NOW,UID=range(1,15)
NODE_FIELDS=('watchdog_mode','watchdog_timeout','telemetry_ms','board_filter_shift','ntc_beta','ntc_nominal_ohms','ntc_pullup_ohms','ntc_nominal_mC')
MOTOR_FIELDS=('control_mode','feedback_source','report_mode','current_limit_mode','current_vref_mv','lower_limit','upper_limit','calibration_min','calibration_max','adc_filter_shift','safe_state','travel_limits_enabled','output_reversed','normalized_limits')
RESULTS=('OK','format','range/mode','disarmed','hardware','no saved configuration','reboot authorization','busy/armed')

def migrate_pages(schema, pages):
    if schema not in (2,3,4) or len(pages)!=4:
        raise ValueError('Expected configuration schema 2, 3 or 4')
    expected=[12,14,14,14] if schema==4 else [12]*4
    if any(len(p)!=length for p,length in zip(pages,expected)):
        raise ValueError('Incorrect configuration page lengths')
    result=[list(p) for p in pages]
    if schema!=4:
        for i in range(3):
            result[i+1] += [((pages[0][8]>>i)&1) if schema==3 else 0, ((pages[0][9]>>i)&1) if schema==3 else 0]
        result[0][8:10]=[0,0]
    return result

class Node:
    def __init__(self, port='COM7', node=0, usb=False):
        self.usb=usb; self.node=node; self.seq=0
        self.link=serial.Serial(port,115200,timeout=.1,write_timeout=1) if usb else Canable(port,node)
    def close(self): self.link.close()
    def raw(self, packet, timeout=1):
        if self.usb:
            self.link.reset_input_buffer()
            self.link.write(b'app '+packet.hex().encode()+b'\n')
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                line=self.link.readline().strip()
                if line.startswith(b'APP '):return bytes.fromhex(line[4:].decode())
        else:
            self.link.send(packet,0x500+self.node)
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                result=self.link.receive(max(.01,deadline-time.monotonic()),0x580+self.node)
                if result is not None and len(result)>=5 and result[1]==(packet[1] if len(packet)>1 else 0):return result
        raise TimeoutError('No application response')
    def request(self,op,arg=0,offset=0,count=0,data=b'',allow_error=False):
        self.seq=(self.seq+1)&255
        result=self.raw(bytes((1,self.seq,op,arg,offset,count))+data)
        if len(result)<5 or result[:3]!=bytes((1,self.seq,op)) or result[4]>len(result)-5:
            raise RuntimeError(f'Malformed application response {result.hex()}')
        code=result[3]
        if code and not allow_error:raise RuntimeError(RESULTS[code] if code<len(RESULTS) else str(code))
        return (code,result[5:5+result[4]]) if allow_error else result[5:5+result[4]]
    def uid(self):
        payload=self.request(UID)
        if len(payload)!=12:raise RuntimeError('Invalid MCU UID response length')
        return ''.join(f'{word:08X}' for word in struct.unpack('<3I',payload))
    def page(self,page):
        count=48 if page==0 else 56
        return struct.unpack('<'+str(count//4)+'i',self.request(GET,page,0,count))
    def write_page(self,page,values):
        count=12 if page==0 else 14
        if len(values)!=count:raise ValueError('Incorrect configuration page length')
        return self.request(SET,page,0,count*4,struct.pack('<'+str(count)+'i',*values))
    def set(self,page,field,value):return self.request(SET,page,field*4,4,struct.pack('<i',value))
    def status(self):
        p=self.request(STATUS)
        return dict(uptime=struct.unpack_from('<I',p)[0],node=p[4],node_status=p[5],motor_status=list(p[6:9]),commands=list(struct.unpack_from('<3b',p,9)),samples=struct.unpack_from('<I',p,12)[0],adc_errors=struct.unpack_from('<I',p,16)[0],drops=struct.unpack_from('<I',p,20)[0])
    def sensors(self,page=0):return struct.unpack('<8Ii' if page==0 else '<9i',self.request(SENSORS,page))
    def motor(self,a,b,c):return self.request(MOTOR,data=struct.pack('<3b',a,b,c))
    def reboot(self):
        key=self.request(REBOOT_ARM)
        return self.request(REBOOT_NOW,data=key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',default='COM7');parser.add_argument('--node',type=int,choices=range(8),default=0)
    parser.add_argument('--usb',action='store_true')
    parser.add_argument('command',choices=('uid','version','status','sensors','page','set','defaults','save','load','arm','disarm','motor','reboot'))
    parser.add_argument('values',nargs='*',type=lambda x:int(x,0))
    args=parser.parse_args();v=args.values
    expected=dict(page=(1,),set=(3,),sensors=(0,1),motor=(3,),defaults=(0,1)).get(args.command,(0,))
    if len(v) not in expected:parser.error('Incorrect number of command values; see docs/APPLICATION_PROTOCOL.md')
    n=Node(args.port,args.node,args.usb)
    try:
        if args.command=='uid':print(n.uid())
        elif args.command=='status':print(n.status())
        elif args.command=='version':print(struct.unpack('<4I',n.request(VERSION)))
        elif args.command=='page':print(n.page(*v))
        elif args.command=='set':n.set(*v);print('OK')
        elif args.command=='sensors':print(n.sensors(*v))
        elif args.command=='motor':n.motor(*v);print('OK')
        elif args.command=='reboot':n.reboot();print('OK reboot queued')
        else:n.request(dict(defaults=DEFAULTS,save=SAVE,load=LOAD,arm=ARM,disarm=DISARM)[args.command],v[0] if v else (255 if args.command=='defaults' else 0));print('OK')
    finally:n.close()
if __name__=='__main__':main()
