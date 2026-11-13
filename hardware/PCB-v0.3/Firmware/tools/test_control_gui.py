"""Control-console tests: fake protocol device + optional physical Hi-Z exercise."""
import argparse
import copy
import json
import queue
import struct
import threading
import time
import tkinter as tk
from pathlib import Path
from krabby_control_gui import *
from krabby import UID

class FakeNode:
    def __init__(self,*args,**kwargs):
        schema=json.loads((ROOT/'docs/config_schema.json').read_text())
        self.cfg=[]
        for page in range(4):
            values=[0]*(12 if page==0 else 14)
            for field in schema['node' if page==0 else 'motors']:values[field['id']]=field['default']
            self.cfg.append(values)
        self.saved=copy.deepcopy(self.cfg)
        self.armed=False;self.commands=(0,0,0);self.motors=[];self.closed=False;self.calls=[]
        self.usb=False
    def request(self,op,arg=0,offset=0,count=0,data=b''):
        self.calls.append(op)
        if op==VERSION:return struct.pack('<4I',1,4,0x08003000,63)
        if op==UID:return struct.pack('<3I',0x12345678,0x9ABCDEF0,0x13579BDF)
        if op==ARM:self.armed=True;self.commands=(0,0,0)
        elif op==DISARM:self.armed=False;self.commands=(0,0,0)
        elif op==SET:self.cfg[arg]=list(struct.unpack('<'+str(len(data)//4)+'i',data))
        elif op==SAVE:self.saved=copy.deepcopy(self.cfg)
        elif op==LOAD:self.cfg=copy.deepcopy(self.saved)
        elif op==DEFAULTS:pass
        elif op==REBOOT_ARM:return b'BBRK'
        elif op==REBOOT_NOW:
            if data!=b'BBRK':raise RuntimeError('key')
        return b''
    def write_page(self,p,values):self.request(SET,p,0,len(values)*4,struct.pack("<"+str(len(values))+"i",*values))
    def uid(self):return ''.join(f'{w:08X}' for w in struct.unpack('<3I',self.request(UID)))
    def page(self,p):return tuple(self.cfg[p])
    def set(self,p,f,v):self.cfg[p][f]=v
    def status(self):return dict(node=0,uptime=1,node_status=9 if self.armed else 1,motor_status=[65]*3,commands=list(self.commands),adc_errors=0,drops=0)
    def sensors(self,p):return (0,)*9
    def motor(self,*values):
        if not self.armed:raise RuntimeError('disarmed')
        self.commands=values;self.motors.append((time.monotonic(),values))
    def raw(self,p):return bytes((1,1,2,0,0))
    def reboot(self):self.request(REBOOT_ARM);self.request(REBOOT_NOW,data=b'BBRK')
    def drain(self):return []
    def close(self):self.closed=True


def wait(predicate,timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.002)
    raise AssertionError('Timed out waiting for console operation')


def synthetic_tests():
    from krabby import migrate_pages
    legacy=[[0]*12 for _ in range(4)]
    legacy[0][8:10]=[5,2]
    migrated=migrate_pages(3,legacy)
    assert migrated[0][8:10]==[0,0]
    assert [p[12:14] for p in migrated[1:]]==[[1,0],[0,1],[1,0]]
    assert [len(p) for p in migrated]==[12,14,14,14]
    legacy[0][8:10]=[0,0]
    assert all(p[12:14]==[0,0] for p in migrate_pages(2,legacy)[1:])
    assert migrate_pages(4,migrated)==migrated
    assert FakeNode().uid()=='123456789ABCDEF013579BDF'
    events=[];f=FakeNode();c=Controller('FAKE',0,False,200,events.append)
    assert repeat_period(20,25)==.02 and repeat_period(20,5)==.0125 and repeat_period(20,1)==.0025
    assert repeat_period(20,0)==.02
    for invalid in (0,1001):
        try:repeat_period(invalid,25)
        except ValueError:pass
        else:raise AssertionError('invalid period accepted')
    c.action(f,'arm_repeat',(20,));assert f.commands==(0,0,0) and c.repeating
    c.action(f,'motor',((127,-127,1),));assert f.commands==(127,-127,1)
    c.action(f,'field',(1,4,123));assert not c.repeating and not f.armed and f.cfg[1][4]==123
    c.action(f,'page',(2,{4:456,10:1}));assert f.cfg[2][4]==456 and f.cfg[2][10]==1
    c.action(f,'arm_repeat',(20,))
    c.action(f,'motor_config',(1,True,True,1000,60000))
    assert not f.armed and not c.repeating
    assert f.cfg[0][8:10]==[0,0] and f.cfg[2][12:14]==[1,1] and f.cfg[2][5:7]==[1000,60000]
    c.action(f,'motor_config',(0,True,False,100,2000))
    assert f.cfg[1][12:14]==[1,0] and f.cfg[2][12:14]==[1,1] and f.cfg[1][5:7]==[100,2000]
    for lower,upper in ((-1,100),(1,65536),(10,10)):
        try:c.action(f,'motor_config',(0,False,True,lower,upper))
        except ValueError:pass
        else:raise AssertionError('invalid normalized limits accepted')
    c.action(f,'save',());c.action(f,'field',(2,4,100));c.action(f,'load',());assert f.cfg[2][4]==456
    c.action(f,'defaults',(255,));c.action(f,'version',());c.action(f,'refresh',())
    c.action(f,'reboot_arm',());c.action(f,'reboot_now',());c.stop.clear()
    c.action(f,'reboot',());c.stop.clear();c.action(f,'raw',('01 01 02 00 00 00',))
    # A GUI motor job queued before a subsequent ARM must never become output.
    c.action(f,'arm',());old_ticket=c.generation
    c.action(f,'arm',())
    try:c.action(f,'motor',((127,127,127),old_ticket))
    except RuntimeError:pass
    else:raise AssertionError('stale pre-ARM GUI command accepted')
    c.action(f,'arm_repeat',(20,));c.last_tx=time.monotonic()-1
    try:c.drive(f)
    except RuntimeError:pass
    else:raise AssertionError('deadline violation accepted')
    c.disarm(f);assert not c.repeating and not f.armed
    # Real scheduling thread, synthetic target. Emergency purge prevents queued ARM.
    target=FakeNode();worker=Controller('FAKE',0,False,100,events.append,factory=lambda *a,**k:target)
    worker.thread.start();wait(lambda:any(e[0]=='connected' for e in events))
    worker.submit('arm_repeat',20);wait(lambda:len(target.motors)>=5)
    worker.halt();wait(lambda:not target.armed and not worker.repeating)
    before=len(target.motors);time.sleep(.05);assert len(target.motors)==before
    worker.stop.set();worker.thread.join(3);assert not worker.thread.is_alive() and target.closed
    # UI rendered without any serial connection; readonly file untouched.
    root=tk.Tk();root.withdraw()
    gui=ControlGUI(root,argparse.Namespace(port='COM7',node=0,usb=False))
    gui.render(f.status(),[f.sensors(p) for p in range(4)],[f.page(p) for p in range(4)],time.monotonic(),time.monotonic())
    gui.staged[1][4]=123;gui.paint_edits();assert gui.config_tables[1].item('4','values')[3]=='123 *'
    assert len(gui.main_tabs.tabs())==5
    root.update();gui.close();root.update()
    if root.winfo_exists():root.destroy()
    return dict(synthetic='PASS',features=['period selection','ARM neutral repeat','signed commands','field/page writes','per-motor polarity/units atomic writes','schema 2/3 JSON migration','save/load/defaults','reboot stages','raw packets','deadline detection','priority stop','worker close','GUI staged edits'])


def physical_test(port,usb,node_id=0):
    # Only proceed with an already disarmed, Hi-Z, CRC-configured fixture.
    direct=Node(port,node_id,usb)
    original_ram=None;saved=None
    try:
        assert not direct.status()['node_status']&8,'Board must already be disarmed'
        original_ram=[direct.page(p) for p in range(4)]
        assert all(original_ram[p][10]==1 for p in (1,2,3)),'All safe states must already be Hi-Z'
        direct.request(LOAD)
        saved=[direct.page(p) for p in range(4)]
        assert all(saved[p][10]==1 for p in (1,2,3)),'Saved configuration must also select Hi-Z'
    finally:
        if original_ram is not None:
            # Firmware SET forces sleep; no pre-DISARM that could wake brake policy.
            for p,values in enumerate(original_ram):direct.write_page(p,values)
        direct.close()
    events=queue.Queue();c=Controller(port,node_id,usb,200,events.put)
    c.thread.start()
    def acknowledged(name):
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            e=events.get(timeout=4)
            if e[0]=='error':raise AssertionError(e)
            if e[0]=='message' and 'failed:' in e[1]:raise AssertionError(e)
            if e[0]=='message' and e[1]==name.upper()+' acknowledged':return
        raise AssertionError(name)
    try:
        while events.get(timeout=5)[0]!='connected':pass
        c.submit('field',0,1,25);acknowledged('field')
        c.submit('arm_repeat',20);acknowledged('arm_repeat')
        time.sleep(.6)
        assert c.repeating,'Stream stopped unexpectedly'
        c.submit('motor',(0,0,0));acknowledged('motor')
        c.halt();time.sleep(.1);assert not c.repeating
        profiles=[]
        for watchdog in (5,1,0):
            c.submit('field',0,1,watchdog);acknowledged('field')
            c.submit('arm_repeat',20);acknowledged('arm_repeat')
            time.sleep(.3)
            profiles.append(dict(watchdog_ms=watchdog*10,period_ms=c.period*1000,
                                 repeated_for_test=c.repeating,max_gap_ms=c.max_gap*1000))
            c.halt();time.sleep(.05)
        c.submit('field',0,1,25);acknowledged('field')
        changed=original_ram[1][4]+1 if original_ram[1][4]<3000 else 2999
        c.submit('field',1,4,changed);acknowledged('field')
        c.submit('save');acknowledged('save')
        c.submit('load');acknowledged('load')
        # Restore saved dataset exactly, then restore original unsaved RAM.
        for p,values in enumerate(saved):c.submit('page',p,dict(enumerate(values)));acknowledged('page')
        c.submit('save');acknowledged('save')
        for p,values in enumerate(original_ram):c.submit('page',p,dict(enumerate(values)));acknowledged('page')
        return dict(physical='PASS',port=port,node=node_id,usb=usb,zero_repeat_seconds=.6,maximum_tx_gap_ms=c.max_gap*1000,watchdog_profiles=profiles,save_load='PASS',restored_original_RAM_and_saved_configuration=True)
    finally:
        c.stop.set();c.thread.join(4)
        assert not c.thread.is_alive()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hardware');parser.add_argument('--usb',action='store_true');parser.add_argument('--node',type=int,default=0,choices=range(8))
    args=parser.parse_args();results=synthetic_tests();print(results)
    previous=ROOT/'bench/application/control_gui_test_results.json'
    if previous.exists():
        results['physical_runs']=json.loads(previous.read_text()).get('physical_runs',[])
    if args.hardware:
        physical=physical_test(args.hardware,args.usb,args.node)
        previous=ROOT/'bench/application/control_gui_test_results.json'
        history=json.loads(previous.read_text()).get('physical_runs',[]) if previous.exists() else []
        results['physical_runs']=history+[physical];print(physical)
    if args.hardware:
        (ROOT/'bench/application/control_gui_test_results.json').write_text(json.dumps(results,indent=2)+'\n')
