"""Jog GUI tests. Synthetic cancellation/routing; optional physical ZERO-only jog."""
import argparse
import hashlib
import json
import struct
import threading
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from krabby_jog_gui import *
from krabby import Node
from test_control_gui import FakeNode, wait
from krabby_control_gui import ROOT

class FakeJogNode(FakeNode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arm_started=threading.Event();self.arm_continue=threading.Event();self.arm_continue.set()
    def request(self,op,*args,**kwargs):
        if op==DISARM:
            self.motors.append((time.monotonic(),(0,0,0)))
        if op==ARM:
            self.arm_started.set();self.arm_continue.wait(2)
        return super().request(op,*args,**kwargs)


def synthetic():
    # Verify the actual JogNode wire wrapper sends zero BEFORE DISARM.
    wire=object.__new__(JogNode);wire.seq=0;packets=[]
    def raw(packet,timeout=1):
        packets.append(packet)
        return bytes((1,packet[1],packet[2],0,0))
    wire.raw=raw
    wire.request(DISARM)
    assert [p[2] for p in packets]==[MOTOR,DISARM] and packets[0][6:]==b'\0\0\0'
    # Every channel and direction has its own three-motor snapshot.
    for motor in range(3):
        for direction in (-1,1):
            f=FakeJogNode();events=[]
            c=JogController('fake',0,False,200,events.append,factory=lambda *a,**k:f)
            c.thread.start();wait(lambda:any(e[0]=='connected' for e in events))
            c.press(motor,direction,-45)
            expected=tuple(direction*45 if i==motor else 0 for i in range(3))
            wait(lambda:f.commands==expected)
            time.sleep(.05);assert c.repeating
            c.release();wait(lambda:not f.armed and not c.repeating)
            assert f.commands==(0,0,0)
            assert f.motors[-1][1]==(0,0,0)
            count=len(f.motors);time.sleep(.04);assert len(f.motors)==count
            c.stop.set();c.thread.join(3);assert not c.thread.is_alive()
    # A click released before the worker starts must never ARM.
    f=FakeJogNode();c=JogController('fake',0,False,200,lambda e:None,factory=lambda *a,**k:f)
    c.press(0,1,80);c.release();c.thread.start();time.sleep(.08);c.stop.set();c.thread.join(3)
    assert ARM not in f.calls
    # Release while ARM is in flight must never emit nonzero drive afterward.
    f=FakeJogNode();f.arm_continue.clear();events=[]
    c=JogController('fake',0,False,200,events.append,factory=lambda *a,**k:f)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events));c.press(1,1,100)
    assert f.arm_started.wait(2)
    c.release();f.arm_continue.set();wait(lambda:DISARM in f.calls and not f.armed and not c.repeating)
    time.sleep(.05);assert all(values==(0,0,0) for _,values in f.motors)
    c.stop.set();c.thread.join(3)
    # Close while held must not retain drive permission.
    f=FakeJogNode();events=[]
    c=JogController('fake',0,False,200,events.append,factory=lambda *a,**k:f)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events));c.press(2,-1,60)
    wait(lambda:f.commands==(0,0,-60));c.stop.set();c.thread.join(3)
    assert not f.armed and f.commands==(0,0,0) and f.closed
    # An ARM rejection cancels the hold; it cannot silently re-arm later.
    class RejectArm(FakeJogNode):
        def request(self,op,*args,**kwargs):
            if op==ARM:raise RuntimeError('hardware inhibit')
            return super().request(op,*args,**kwargs)
    f=RejectArm();events=[]
    c=JogController('fake',0,False,200,events.append,factory=lambda *a,**k:f)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events));token=c.press(0,1,80)
    wait(lambda:any(e[0]=='message' and 'failed:' in e[1] for e in events))
    assert not c.is_held(token) and not c.repeating and not f.armed
    c.stop.set();c.thread.join(3)
    # Existing control/read-only GUIs are imported, never edited.
    root=tk.Tk();root.withdraw();gui=JogGUI(root,argparse.Namespace(port='COM7',node=7,usb=False))
    assert gui.main_tabs.tab(gui.jog_tab,'text')=='Jog' and len(gui.jog_drives)==3
    root.update();root.destroy()
    return dict(synthetic='PASS',cases=['all six directions','signed magnitude','hold repeats','release zeros/disarms','no restart','release before ARM','release during ARM','zero-before-DISARM wire order','Jog tab widgets','close while held','ARM rejection cancels hold'])


def physical(port,node,usb):
    probe=Node(port,node,usb)
    try:
        before=[probe.page(p) for p in range(4)]
        assert not probe.status()['node_status']&8
        assert all(before[p][10]==1 for p in (1,2,3)), 'Physical test requires existing Hi-Z safe states'
    finally:probe.close()
    events=[];c=JogController(port,node,usb,200,events.append)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events),5)
    runs=[]
    try:
        for motor in range(3):
            for direction in (-1,1):
                token=c.press(motor,direction,0) # NO energized movement in unattended test.
                wait(lambda:c.active_jog==token and c.repeating)
                time.sleep(.1)
                c.release();wait(lambda:not c.repeating and c.active_jog is None)
                wait(lambda:any(e[0]=='message' and e[1]=='STOP / DISARM acknowledged' for e in events))
                events.clear()
                runs.append(dict(motor=motor+1,direction=direction,drive=0))
    finally:
        c.release();c.stop.set();c.thread.join(4);assert not c.thread.is_alive()
    probe=Node(port,node,usb)
    try:
        state=probe.status();assert not state['node_status']&8 and state['commands']==[0,0,0]
        assert [probe.page(p) for p in range(4)]==before
    finally:probe.close()
    return dict(interface='USB' if usb else 'CAN',port=port,node=node,zero_only_runs=runs,final_status=state,configuration_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hardware');parser.add_argument('--node',type=int,default=7);parser.add_argument('--usb',action='store_true')
    args=parser.parse_args();record=synthetic();print(record)
    target=ROOT/'bench/application/jog_gui_test_results.json'
    record['physical_runs']=json.loads(target.read_text()).get('physical_runs',[]) if target.exists() else []
    if args.hardware:
        run=physical(args.hardware,args.node,args.usb);record['physical_runs'].append(run);print(run)
    if args.hardware:
        target.write_text(json.dumps(record,indent=2)+'\n')
