"""Physical single-board acceptance, no energized motor stages.
Requires ST-Link and actual COM7/COM3 hardware. Configuration is overwritten.
All motor tests use Hi-Z + travel inhibition; they do not prove motor drive.
"""
import json, struct, subprocess, re, time
from pathlib import Path
from krabby import *
from openblt_canable import APP_BASE,CONFIG_BASE,CONFIG_SIZE,ROOT,update,load_image
from bench_tools import programmer_cli
OUT=ROOT/'bench/application';RESULTS=[]
def record(name,**detail):
    RESULTS.append(dict(test=name,result='PASS',**detail))
    (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2)+'\n')
    print('PASS',name,detail,flush=True)
def stlink(*args):
    p=subprocess.run([programmer_cli(),'-c','port=SWD','mode=UR' if '-w' in args else 'mode=HOTPLUG',*args],capture_output=True,timeout=20)
    text=p.stdout.decode(errors='replace')+p.stderr.decode(errors='replace')
    if p.returncode or 'Error:' in text:raise RuntimeError(text)
    return text
def sleep_pins_low():
    text=stlink('-r32','0x48000414','4','-r32','0x40012C34','12')
    odr=int(re.search(r'0x48000414\s*:\s*([0-9A-Fa-f]+)',text)[1],16)
    assert not odr&((1<<10)|(1<<14)|(1<<15)),text
    return hex(odr)
def main():
    n=Node(); usb=Node('COM3',usb=True)
    try:
        s=n.status();assert not s['node_status']&8;sleep_pins_low()
        text=stlink('-r32','0x40012C44','4')
        assert not int(re.search(r'0x40012C44\s*:\s*([0-9A-Fa-f]+)',text)[1],16)&(1<<15)
        record('Boot disarmed; motor sleeps physically LOW',status=s)
        assert struct.unpack('<4I',n.request(VERSION))==(1,4,APP_BASE,63)
        assert usb.request(VERSION)==n.request(VERSION)
        record('CAN and USB version identity agree')
        # No power-stage energization: configure every safe state Hi-Z first.
        for m in (1,2,3):n.set(m,10,1)
        for page in range(4):
            before_pages=[n.page(i) for i in range(4)]
            field,value=(1,30) if page==0 else (4,120+page)
            n.set(page,field,value)
            assert usb.page(page)==n.page(page)
            assert n.page(page)[field]==value
            for other in range(4):
                if other!=page:assert usb.page(other)==n.page(other)==before_pages[other]
            record('RAM configuration page read/write across CAN and USB',page=page)
        for page,field,value in ((0,0,1),(0,1,256),(0,2,1),(0,3,9),(0,8,1),(1,0,1),(1,1,2),(1,2,2),(1,3,1),(1,4,0),(1,4,3001),(1,9,9),(1,10,2),(1,11,2)):
            before=n.page(page)
            code,_=n.request(SET,page,field*4,4,struct.pack('<i',value),allow_error=True)
            assert code==2 and n.page(page)==before
        record('Reserved modes, invalid bounds, and fields rejected atomically',cases=14)
        for page in range(4):
            size=48 if page==0 else 56
            for off,count in ((1,4),(size,4),(size-4,8),(0,0),(0,size+1)):
                assert n.request(GET,page,off,count,allow_error=True)[0]==2
        record('Page access boundaries validated',cases=20)
        # Whole-page writes and per-page defaults use the same validation/services.
        page=list(n.page(2));page[4]=145
        n.write_page(2,page);assert n.page(2)==tuple(page)
        others=[n.page(i) for i in range(4)];saved_flag=n.status()['node_status']&2
        n.request(DEFAULTS,1);assert n.page(1)[4]==100
        for other in (0,2,3):assert n.page(other)==others[other]
        assert n.status()['node_status']&2==saved_flag;n.set(1,10,1)
        record('Whole-page SET and isolated defaults preserve other pages and saved-image status')
        # ADC physical sampling and native sensors, without external stimulus.
        before=n.status();time.sleep(.5);after=usb.status()
        assert after['samples']-before['samples']>400 and after['adc_errors']==before['adc_errors']==0
        for page in range(4):
            a=n.sensors(page);assert len(a)==9
            record('Native physical ADC/encoder snapshot',page=page,values=a)
        assert n.sensors(0)[8]==-2147483648
        n.set(0,4,3950)
        import math
        snapshot=n.sensors(0)
        expected_temp=(1/(1/298.15+math.log(snapshot[5]/10000)/3950)-273.15)*1000
        assert abs(snapshot[8]-expected_temp)<150
        n.set(0,4,0)
        record('NTC uncalibrated sentinel and configured beta math; not temperature calibration')
        # Inject counters only, keep drivers asleep. Not an external encoder stimulus.
        for motor,address,delta in ((1,0x40000024,100),(2,0x40013424,-100),(3,0x40000C24,100)):
            n.set(motor,1,1);stlink('-w32',hex(address),'0');time.sleep(.03);before=n.sensors(motor)[5]
            stlink('-w32',hex(address),'100');time.sleep(.03)
            after=n.sensors(motor)[5]
            assert after-before==delta,(motor,before,after)
            record('Encoder register injection, not physical motion',motor=motor,delta=delta)
            n.set(motor,1,0)
        for motor in (1,2,3):
            n.set(motor,2,1);p=n.sensors(motor);assert 0<=p[6]<=65535
            n.set(motor,2,0)
        record('Normalized/native feedback selection independent per motor')
        # Hardware DAC command acceptance, not measured analog outputs.
        codes=[]
        for motor,mv in ((1,100),(2,200),(3,300)):
            n.set(motor,4,mv);code=n.sensors(motor)[7]
            expected=(mv*4096*101+165000)//330000
            assert code==expected,(code,expected);codes.append(code)
        record('Independent physical I2C DAC writes acknowledged; codes reported',codes=codes)
        # Block both command directions using railed disconnected pots or tight limits.
        for motor in (1,2,3):
            pot=n.sensors(motor)[4]
            # On this unconnected fixture native pots are railed. Require this
            # before issuing any nonzero request; do not guess fixture safety.
            assert pot in (0,4095),('Cannot safely run inhibited output tests',motor,pot)
            n.set(motor,11,1)
        n.set(0,1,0)
        assert n.request(MOTOR,data=bytes((127,127,127)),allow_error=True)[0]==3
        n.request(ARM);s=n.status();assert s['node_status']&8 and s['commands']==[0,0,0]
        assert n.request(SAVE,allow_error=True)[0]==7
        assert n.request(LOAD,allow_error=True)[0]==7
        record('ARM clears commands; save/load reject while armed')
        for values in ((127,-127,0),(0,0,0),(-127,127,-127)):
            n.motor(*values);s=n.status();assert s['commands']==list(values)
            assert all(not x&2 for x in s['motor_status']);sleep_pins_low()
        record('Signed commands accepted, travel inhibition and Hi-Z verified on GPIO')
        # A USB transaction started before a new CAN ARM cannot become a fresh command.
        n.request(DISARM)
        usb.link.reset_input_buffer();usb.link.write(b'app 01010b0000007f7f7f')
        time.sleep(.03);n.request(ARM);usb.link.write(b'\n')
        deadline=time.monotonic()+1;line=b''
        while time.monotonic()<deadline:
            line=usb.link.readline().strip()
            if line.startswith(b'APP '):break
        assert bytes.fromhex(line[4:].decode())[3]==3,line
        assert n.status()['commands']==[0,0,0]
        n.motor(0,0,0)
        record('Pre-ARM USB command rejected across CAN ARM; fresh transaction accepted')
        # Irrelevant queries and malformed motor packets cannot refresh watchdog.
        n.set(0,1,5);n.request(ARM);n.motor(127,127,127)
        start=time.monotonic()
        while time.monotonic()-start<.12:
            n.request(STATUS)
        s=n.status();assert not s['node_status']&8 and s['node_status']&4 and s['commands']==[0,0,0]
        assert n.request(MOTOR,data=bytes((1,2,3)),allow_error=True)[0]==3
        sleep_pins_low();record('Watchdog timeout disarms, clears commands; queries cannot refresh',status=s)
        n.request(ARM);assert n.status()['commands']==[0,0,0]
        assert n.request(MOTOR,data=bytes((128,0,0)),allow_error=True)[0]==2
        time.sleep(.08);assert not n.status()['node_status']&8
        record('Rearm requires fresh command; -128 rejected without watchdog refresh')
        n.set(0,1,255);n.request(ARM);n.motor(127,0,0)
        time.sleep(.07);assert n.status()['node_status']&8
        n.request(DISARM);assert n.status()['commands']==[0,0,0]
        n.set(0,1,0);n.request(ARM);time.sleep(.1);assert n.status()['node_status']&8
        n.request(DISARM);n.set(0,1,25)
        record('Watchdog 0 disables and 255 supported; explicit DISARM clears')
        # 50 Hz complete motor snapshots; all outputs still locally inhibited.
        n.request(ARM)
        stream_start=time.monotonic()
        for i in range(60):
            n.motor(127,-127,i%127);time.sleep(max(0,stream_start+(i+1)*.02-time.monotonic()))
        assert n.status()['node_status']&8;n.request(DISARM);sleep_pins_low()
        record('Physical 50 Hz motor stream with inhibited outputs',frames=60)
        # Malformed/version/wrong-node frames, while disarmed.
        for packet in (b'',b'\x01',b'\x02\x01\x02\0\0\0',b'\x01\x02\x0b\0\0\0\x01',b'\x01\x03\x02\0\0\0\x01'):
            result=n.raw(packet)
            assert result[3]==1
        assert n.request(99,allow_error=True)[0]==2
        n.link.send(bytes((1,50,2,0,0,0)),0x501)
        assert n.link.receive(.1,0x581) is None
        record('Malformed packets, version mismatch, unknown operation, wrong node')
        # Telemetry does not require USB. Validate rotating sensor/status types.
        n.set(0,2,10);seen=set();deadline=time.monotonic()+.3
        while time.monotonic()<deadline:
            p=n.link.receive(.05,0x600)
            if p:seen.add((p[1],p[2]))
        assert {(0,2),(1,3),(2,3),(3,3),(4,3)}<=seen,seen
        n.set(0,2,0);record('Periodic status and every sensor page observed over CAN',types=sorted(seen))
        # Reboot cannot happen with missing/wrong/expired/cross-interface permission.
        assert n.request(REBOOT_NOW,data=bytes(4),allow_error=True)[0]==6
        key=n.request(REBOOT_ARM)
        assert usb.request(REBOOT_NOW,data=key,allow_error=True)[0]==6
        key=n.request(REBOOT_ARM);time.sleep(.55)
        assert n.request(REBOOT_NOW,data=key,allow_error=True)[0]==6
        record('Reboot missing, cross-transport, and expired permissions rejected')
        # Valid saved configuration, reset round-trip, and RAM isolation.
        expected=[n.page(i) for i in range(4)]
        n.request(SAVE);assert not n.status()['node_status']&2
        n.set(1,4,111);assert not n.status()['node_status']&2
        n.request(LOAD);assert [n.page(i) for i in range(4)]==expected
        before=n.status()['uptime'];n.reboot();time.sleep(2.7)
        assert n.status()['uptime']<before and not n.status()['node_status']&8
        assert [n.page(i) for i in range(4)]==expected
        record('SAVE/LOAD and two-stage reboot preserve configuration, boot disarmed')
        # Physical CAN firmware update preserves valid application-managed config.
        usb.close();n.close();a=Canable()
        try:
            a.connect();saved=a.read(CONFIG_BASE,CONFIG_SIZE)
            assert a.read(0x08000000,APP_BASE-0x08000000)==(OUT/'boot_before.bin').read_bytes()
            update(a,ROOT/'build/Debug/krabby_v03_bringup.srec',start=False)
            assert a.read(CONFIG_BASE,CONFIG_SIZE)==saved
            # Abandon an actual update before vector finalization, then reset
            # beyond boot-entry timeout and recover using CAN programming only.
            assert update(a,ROOT/'build/Debug/krabby_v03_bringup.srec',start=False,interrupt_after=1024) is False
            stlink('-rst');time.sleep(2.3);a.connect()
            assert a.read(APP_BASE,8)==bytes([255])*8
            assert a.read(CONFIG_BASE,CONFIG_SIZE)==saved
            assert a.read(0x08000000,APP_BASE-0x08000000)==(OUT/'boot_before.bin').read_bytes()
            update(a,ROOT/'build/Debug/krabby_v03_bringup.srec',start=False)
            record('Interrupted application update survives reset; CAN-only reprogramming preserves config and boot')
            a.request(b'\xcf',.1)
        finally:a.close()
        time.sleep(.7);n=Node();usb=Node('COM3',usb=True)
        assert [n.page(i) for i in range(4)]==expected and not n.status()['node_status']&2
        record('OpenBLT unchanged; physical CAN update preserved CRC-verified config')
        # Corrupt only config page using ST-Link (no options/protection changes).
        n.close();usb.close()
        corrupt=bytearray(saved);corrupt[16]^=1
        path=OUT/'config_corrupt.bin';path.write_bytes(corrupt)
        stlink('-w',str(path),hex(CONFIG_BASE),'-v');stlink('-rst');time.sleep(2.7)
        n=Node();usb=Node('COM3',usb=True)
        assert n.status()['node_status']&2
        # Factory defaults may brake on DISARM: first set all channels Hi-Z.
        for m in (1,2,3):n.set(m,10,1)
        n.request(ARM);assert n.status()['node_status']&8
        n.request(DISARM)
        assert n.request(LOAD,allow_error=True)[0]==5
        n.request(SAVE);assert not n.status()['node_status']&2
        record('Corrupt saved data gives NO_CONFIG, provisional config still arms; save recovers')
        # Erased and header-incomplete images take the identical NO_CONFIG path.
        for name, fixture in (('erased',bytes([255])*CONFIG_SIZE),('partial-header',bytes(8)+saved[8:])):
            n.close();usb.close();path=OUT/(name+'.bin');path.write_bytes(fixture)
            stlink('-w',str(path),hex(CONFIG_BASE),'-v');stlink('-rst');time.sleep(2.7)
            n=Node();usb=Node('COM3',usb=True)
            assert n.status()['node_status']&2 and not n.status()['node_status']&8
            for motor in (1,2,3):n.set(motor,10,1)
            n.request(ARM);assert n.status()['node_status']&8
            n.request(DISARM);n.request(SAVE)
            record('Invalid saved-image recovery with provisional RAM',fixture=name)
        # Explicit all-page defaults are a separate operation from LOAD.
        n.request(DEFAULTS,255)
        schema=json.loads((ROOT/'docs/config_schema.json').read_text())
        for page in range(4):
            values=n.page(page)
            for field in schema['node' if page==0 else 'motors']:
                assert values[field['id']]==field['default']
        for motor in (1,2,3):n.set(motor,10,1)
        n.request(SAVE)
        record('All factory-default fields match documented schema; bench Hi-Z saved')
        # Leave bench-safe saved configuration, all sleeps LOW.
        sleep_pins_low();record('Final board disarmed, Hi-Z defaults saved',status=n.status())
    finally:
        n.close();usb.close()
    print('PASS',len(RESULTS),'physical application checks')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware',action='store_true',required=True,help='Explicitly allow configuration overwrite and ST-Link operations on the node-0 fixture')
    parser.parse_args()
    main()
