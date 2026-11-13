"""PC position-loop tests: synthetic plant; optional physical zero-limit exercise."""
import argparse,json,struct,time,tkinter as tk
from pathlib import Path
from krabby_position_gui import *
from test_jog_gui import FakeJogNode
from test_control_gui import wait
from krabby import Node
from krabby_control_gui import ROOT

class Plant(FakeJogNode):
    def __init__(self,*a,**k):
        super().__init__(*a,**k);self.pots=[1000,2000,3000];self.bad=False
    def sensors(self,page):
        if self.bad:raise TimeoutError('feedback timeout')
        if not page:return (0,)*9
        i=page-1;pot=self.pots[i]
        return (0,0,0,pot,pot,0,pot,125,abs(self.commands[i])*10)
    def motor(self,*values):
        super().motor(*values)
        for i,duty in enumerate(values):self.pots[i]=max(0,min(4095,self.pots[i]+duty*4))


def synthetic():
    cfg=[0]*14;cfg[7]=100;cfg[8]=1100
    assert normalized_position([0,0,0,0,600,123,999],cfg)==32767
    cfg[1]=1;assert normalized_position([0,0,0,0,600,1100,0],cfg)==65535
    assert position_duty(65535,0,20,400,200)==20
    assert position_duty(0,65535,127,400,200)==-127
    assert position_duty(1000,900,20,400,200)==0
    assert position_duty(65535,0,0,400,200)==0
    plant=Plant();events=[]
    c=PositionController('fake',0,False,200,events.append,factory=lambda *a,**k:plant)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events))
    c.settings(None,[10,15,0],400,200)
    token=c.enable_position();wait(lambda:c.position_active==token)
    time.sleep(.08);assert all(v==(0,0,0) for _,v in plant.motors)
    targets=list(c.targets);targets[0]=50000;targets[1]=20000
    c.settings(targets,[10,15,0],400,200)
    wait(lambda:abs(plant.pots[0]*65535//4095-50000)<=200 and abs(plant.pots[1]*65535//4095-20000)<=200,3)
    assert all(abs(v[0])<=10 and abs(v[1])<=15 and v[2]==0 for _,v in plant.motors)
    c.disable_position();wait(lambda:not plant.armed and not c.repeating)
    assert plant.commands==(0,0,0)
    # Cancel while ARM is in flight, with no delayed motion.
    plant.arm_started.clear();plant.arm_continue.clear()
    token=c.enable_position();assert plant.arm_started.wait(2)
    c.disable_position();plant.arm_continue.set();time.sleep(.1)
    assert not plant.armed and not c.repeating
    # Invalid active feedback stops and requires fresh enable.
    plant.pots[0]=0
    token=c.enable_position();wait(lambda:any(e[0]=='message' and 'ADC rail' in e[1] for e in events))
    assert not plant.armed and not c.repeating
    c.stop.set();c.thread.join(3);assert not c.thread.is_alive()
    # GUI additions and normalized (not firmware report-mode dependent) readouts.
    root=tk.Tk();root.withdraw()
    gui=PositionGUI(root,argparse.Namespace(port='COM7',node=7,usb=False))
    f=Plant();now=time.monotonic()
    gui.render(f.status(),[f.sensors(i) for i in range(4)],[f.page(i) for i in range(4)],now,now)
    assert gui.main_tabs.tab(gui.position_tab,'text')=='Position control'
    assert str(1000*65535//4095) in gui.position_readouts[0].get()
    assert all(v.get()=='0' for v in gui.position_limits)
    # Board configuration is plain text, refreshed even while position control is active.
    cfg=[list(f.page(i)) for i in range(4)]
    cfg[1][5:7]=[1200,50000];cfg[1][11:14]=[1,1,1]
    state=f.status();state['motor_status'][0]=4
    gui.position_ui_requested=True;gui.position_starting=False
    gui.render(state,[f.sensors(i) for i in range(4)],cfg,now,now)
    text=gui.board_config_readouts[0].get()
    assert all(part in text for part in ('Enabled','Normalized','1200','50000','Output reversed: Yes','Boundary: lower'))
    cfg[1][11:14]=[0,0,0]
    gui.render(state,[f.sensors(i) for i in range(4)],cfg,now,now)
    assert 'Disabled' in gui.board_config_readouts[0].get() and 'Native' in gui.board_config_readouts[0].get()
    assert not hasattr(gui,'write_motor_config') and not hasattr(gui,'position_reverse')
    gui.position_ui_requested=False
    # Initial target acquisition must survive coalescing of many live samples.
    gui.position_ui_requested=True;gui.position_starting=True;gui.position_ui_token=99
    gui.publish(('position_sample',99,True,[10000,20000,30000],[0,0,0],(0,0,0)))
    for _ in range(20):gui.publish(('position_sample',99,False,[11000,21000,31000],[123,456,789],(0,0,0)))
    gui.pump()
    assert [int(v.get()) for v in gui.position_targets]==[10000,20000,30000] and not gui.position_starting
    assert [v.get() for v in gui.current_readouts]==[f'Measured current: {v} mA' for v in (123,456,789)]
    gui.position_ui_requested=False
    root.update();root.destroy()
    return dict(synthetic='PASS',tests=['native pot/encoder normalization','signed P control','deadband','speed caps','firmware owns direction polarity','enable neutral until target moved','two-motor simulated plant convergence','disable zeros/disarms','cancel during ARM','ADC rail inhibits','GUI normalized/current readouts'])


def physical(port,node_id,usb):
    probe=Node(port,node_id,usb)
    try:
        before=[probe.page(i) for i in range(4)]
        assert not probe.status()['node_status']&8
        assert all(before[i][10]==1 for i in (1,2,3))
    finally:probe.close()
    events=[];c=PositionController(port,node_id,usb,200,events.append)
    c.thread.start();wait(lambda:any(e[0]=='connected' for e in events),5)
    try:
        c.settings(None,[0,0,0],400,200)
        token=c.enable_position();wait(lambda:c.position_active==token)
        time.sleep(.25)
        c.settings([65535,32767,0],[0,0,0],400,200)
        time.sleep(.25);assert c.repeating and c.values==(0,0,0)
        samples=[e for e in events if e[0]=='position_sample']
        assert len(samples)>=5
        c.disable_position();wait(lambda:not c.repeating and c.position_active is None)
    finally:c.stop.set();c.thread.join(4);assert not c.thread.is_alive()
    probe=Node(port,node_id,usb)
    try:
        state=probe.status();assert not state['node_status']&8 and state['commands']==[0,0,0]
        assert [probe.page(i) for i in range(4)]==before
    finally:probe.close()
    return dict(port=port,node=node_id,zero_speed_limits=True,position_samples=len(samples),positions=samples[-1][3],currents_mA=samples[-1][4],max_motor_gap_ms=c.max_gap*1000,configuration_unchanged=True,final_status=state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hardware');parser.add_argument('--node',type=int,default=7);parser.add_argument('--usb',action='store_true')
    args=parser.parse_args();results=synthetic();print(results)
    path=ROOT/'bench/application/position_gui_test_results.json'
    results['physical_runs']=json.loads(path.read_text()).get('physical_runs',[]) if path.exists() else []
    if args.hardware:
        run=physical(args.hardware,args.node,args.usb);results['physical_runs'].append(run);print(run)
    if args.hardware:
        path.write_text(json.dumps(results,indent=2)+'\n')
