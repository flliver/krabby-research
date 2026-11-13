"""Krabby PC-side proportional position loop. Firmware owns motor polarity and travel limits."""
import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from krabby import ARM
from krabby_control_gui import repeat_period
from krabby_jog_gui import JogGUI, JogController


def normalized_position(sensor, config):
    low, high = config[7], config[8]
    if high <= low:
        raise ValueError('Calibration maximum must exceed minimum')
    native = sensor[5] if config[1] else sensor[4]
    return max(0, min(65535, (native-low)*65535//(high-low)))


def position_duty(target, position, maximum, gain, deadband):
    error = target-position
    if maximum == 0 or abs(error) <= deadband:
        return 0
    duty = min(maximum, max(1, int(abs(error)*gain/65535)))
    return duty * (-1 if error < 0 else 1)


class PositionController(JogController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_lock = threading.Lock()
        self.position_token = 0
        self.position_requested = False
        self.position_active = None
        self.targets = [0,0,0]
        self.target_live = [False]*3
        self.caps = [0]*3
        self.gain = 400.0
        self.deadband = 200

    def settings(self, targets, caps, gain, deadband):
        if len(caps)!=3 or (targets is not None and len(targets)!=3):
            raise ValueError('Three motor settings required')
        if any(type(v) is not int or not 0 <= v <= 127 for v in caps):
            raise ValueError('Speed limits must be integer 0..127')
        if targets is not None and any(type(v) is not int or not 0 <= v <= 65535 for v in targets):
            raise ValueError('Targets must be integer 0..65535')
        if not 0 < gain <= 10000 or not 0 <= deadband <= 65535:
            raise ValueError('Gain must be 0..10000 (nonzero); deadband 0..65535')
        with self.position_lock:
            self.caps = list(caps)
            self.gain, self.deadband = gain, deadband
            if targets is not None:
                for i,value in enumerate(targets):
                    if value != self.targets[i]:
                        self.target_live[i] = True
                self.targets = list(targets)

    def enable_position(self):
        self.halt()
        with self.position_lock:
            self.position_token += 1
            token = self.position_token
            self.position_requested = True
        self.submit('position_start', token)
        return token

    def disable_position(self):
        with self.position_lock:
            self.position_requested = False
        self.halt()

    def wanted(self, token):
        with self.position_lock:
            return self.position_requested and self.position_token == token

    def disarm(self, node):
        previous=self.position_active
        if previous is not None:
            with self.position_lock:
                if self.position_token==previous:self.position_requested=False
            self.position_active=None
        super().disarm(node)
        if previous is not None:self.publish(('position_off',previous))

    def action(self, node, name, args):
        if name != 'position_start':
            if name not in ('version','refresh'):
                with self.position_lock:
                    self.position_requested = False
                if self.position_active is not None:
                    self.disarm(node)
            return super().action(node, name, args)
        token = args[0]
        try:
            if not self.wanted(token):
                return
            super().disarm(node)
            self.position_config = [node.page(i) for i in range(4)]
            self.watchdog = self.position_config[0][1]
            self.period = repeat_period(20,self.watchdog)
            sensors = [node.sensors(i) for i in (1,2,3)]
            positions = [normalized_position(s,self.position_config[i+1]) for i,s in enumerate(sensors)]
            with self.position_lock:
                self.targets = list(positions)
                self.target_live = [False]*3
            if not self.wanted(token) or self.stop.is_set():
                return
            node.request(ARM)
            self.generation += 1
            self.arm_permitted = True
            if not self.wanted(token) or self.stop.is_set() or self.urgent.is_set():
                self.disarm(node)
                return
            self.position_active = token
            self.last_tx = None
            self.max_gap = 0
            self.repeating = True
            self.values = (0,0,0)
            self.publish(('position_sample',token,True,positions,[s[8] for s in sensors],self.values))
            self.drive(node)
        except Exception:
            with self.position_lock:
                if self.position_token == token:
                    self.position_requested = False
            self.publish(('position_off',token))
            raise

    def drive(self, node):
        token = self.position_active
        if token is None:
            return super().drive(node)
        if not self.wanted(token) or self.stop.is_set():
            self.disarm(node)
            return
        started = time.monotonic()
        sensors = [node.sensors(i) for i in (1,2,3)]
        with self.position_lock:
            targets,caps,live = list(self.targets),list(self.caps),list(self.target_live)
            gain,deadband = self.gain,self.deadband
        positions=[];commands=[]
        for i,sensor in enumerate(sensors):
            config=self.position_config[i+1]
            position=normalized_position(sensor,config)
            if caps[i] and not config[1] and sensor[3] in (0,4095):
                raise RuntimeError(f'M{i+1} potentiometer at ADC rail; position control stopped')
            positions.append(position)
            commands.append(position_duty(targets[i],position,caps[i],gain,deadband) if live[i] else 0)
        if not self.wanted(token) or self.stop.is_set() or self.urgent.is_set():
            self.disarm(node)
            return
        self.values=tuple(commands)
        super().drive(node)
        self.next_motor=max(time.monotonic(),started+self.period)
        if not self.wanted(token) or self.stop.is_set():
            self.disarm(node)
            return
        self.publish(('position_sample',token,False,positions,[s[8] for s in sensors],tuple(commands)))
class PositionGUI(JogGUI):
    def __init__(self,root,args):
        self.position_tab=None
        self.position_events=queue.Queue(maxsize=4)
        self.position_initial=queue.Queue(maxsize=1)
        self.position_ui_requested=False
        self.position_starting=False
        self.position_ui_token=None
        super().__init__(root,args)
        root.title('Krabby control console v3 — PC position loop')
        frame=ttk.Frame(self.main_tabs,padding=15)
        self.main_tabs.add(frame,text='Position control')
        self.position_tab=str(frame)
        self.position_enable=tk.BooleanVar(value=False)
        ttk.Checkbutton(frame,text='Enable PC closed-loop position control',variable=self.position_enable,command=self.toggle_position).grid(row=0,column=0,columnspan=5,sticky='w',pady=8)
        ttk.Label(frame,text='Enable initializes targets to measured positions. Move a slider to command motion. Speed 0 disables that motor; Apply/Enter commits speed edits.').grid(row=1,column=0,columnspan=5,sticky='w',pady=5)
        self.position_targets=[tk.DoubleVar(value=0) for _ in range(3)]
        self.position_limits=[tk.StringVar(value='0') for _ in range(3)]
        self.board_config_readouts=[tk.StringVar(value='Board configuration: awaiting sample') for _ in range(3)]
        self.target_text=[tk.StringVar(value='0') for _ in range(3)]
        self.position_readouts=[tk.StringVar(value='Position: -- / 65535') for _ in range(3)]
        self.current_readouts=[tk.StringVar(value='Measured current: -- mA') for _ in range(3)]
        self.output_readouts=[tk.StringVar(value='Last drive: 0') for _ in range(3)]
        for i in range(3):
            box=ttk.LabelFrame(frame,text=f'Motor {i+1}',padding=10)
            box.grid(row=2+i,column=0,columnspan=5,sticky='ew',pady=6)
            box.columnconfigure(0,weight=1)
            ttk.Scale(box,from_=0,to=65535,variable=self.position_targets[i],command=lambda value,m=i:self.target_changed(m,value)).grid(row=0,column=0,columnspan=3,sticky='ew',padx=5)
            ttk.Label(box,textvariable=self.target_text[i],width=8).grid(row=0,column=3)
            ttk.Label(box,text='Max drive (0–127)').grid(row=1,column=0,sticky='w')
            speed=ttk.Spinbox(box,from_=0,to=127,textvariable=self.position_limits[i],width=8,command=self.post_settings)
            speed.grid(row=1,column=1)
            speed.bind('<Return>',lambda _e:self.post_settings())
            info=ttk.LabelFrame(box,text='Board configuration (read only)',padding=6)
            info.grid(row=3,column=0,columnspan=4,sticky='ew',pady=(8,0))
            ttk.Label(info,textvariable=self.board_config_readouts[i],justify='left',wraplength=740).pack(anchor='w')
            live=ttk.Frame(box)
            live.grid(row=2,column=0,columnspan=4,sticky='w',pady=(8,0))
            ttk.Label(live,textvariable=self.position_readouts[i]).pack(side='left',padx=(0,20))
            ttk.Label(live,textvariable=self.current_readouts[i],font=('Segoe UI',10,'bold')).pack(side='left',padx=(0,20))
            ttk.Label(live,textvariable=self.output_readouts[i]).pack(side='left')
        self.position_gain=tk.StringVar(value='400')
        self.position_deadband=tk.StringVar(value='200')
        ttk.Label(frame,text='P gain (drive / full normalized travel)').grid(row=5,column=0,sticky='w',pady=8)
        ttk.Entry(frame,textvariable=self.position_gain,width=8).grid(row=5,column=1)
        ttk.Label(frame,text='Deadband (normalized units)').grid(row=5,column=2)
        ttk.Entry(frame,textvariable=self.position_deadband,width=8).grid(row=5,column=3)
        ttk.Button(frame,text='Apply speed / tuning',command=self.post_settings).grid(row=5,column=4,padx=6)
        self.position_state=tk.StringVar(value='Position loop OFF')
        ttk.Label(frame,textvariable=self.position_state).grid(row=6,column=0,columnspan=5,sticky='w')
        ttk.Button(frame,text='STOP / DISARM',command=self.panic).grid(row=7,column=0,sticky='w',pady=8)
        ttk.Label(frame,text='PC-side proportional loop. Edit board limits and polarity only on Configuration > Motor pages; SAVE persists changes. Verify calibration and output sign with Jog first.\nReadouts use configured feedback source and calibration endpoints, normalized locally to 0–65535.\nChanging tabs, losing focus, STOP, disable, disconnect or errors cancels control. No automatic re-arm.').grid(row=8,column=0,columnspan=5,sticky='w')
        root.bind('<FocusOut>',lambda _e:root.after(30,self.position_focus),add='+')
        root.bind('<Unmap>',lambda e:self.position_stop() if e.widget==root else None,add='+')

    def publish(self,event):
        if event[0]=='position_sample' and event[2]:
            try:self.position_initial.put_nowait(event)
            except queue.Full:
                try:self.position_initial.get_nowait()
                except queue.Empty:pass
                self.position_initial.put_nowait(event)
        elif event[0].startswith('position_'):
            try:self.position_events.put_nowait(event)
            except queue.Full:
                try:self.position_events.get_nowait()
                except queue.Empty:pass
                self.position_events.put_nowait(event)
        else:super().publish(event)

    def post_settings(self):
        try:
            caps=[int(v.get()) for v in self.position_limits]
            gain=float(self.position_gain.get());deadband=int(self.position_deadband.get())
            targets=None if self.position_starting else [int(round(v.get())) for v in self.position_targets]
            if isinstance(self.controller,PositionController):
                self.controller.settings(targets,caps,gain,deadband)
            elif any(not 0<=v<=127 for v in caps) or not 0<gain<=10000 or not 0<=deadband<=65535:
                raise ValueError('Invalid drive limit, gain or deadband')
            return True
        except Exception as error:
            self.append_log('Position settings: '+str(error))
            self.position_stop()
            return False

    def target_changed(self,motor,value):
        if hasattr(self,'target_text'):
            self.target_text[motor].set(str(int(round(float(value)))))
        if self.position_ui_requested and not self.position_starting:self.post_settings()

    def toggle_position(self):
        if not self.position_enable.get():self.position_stop();return
        if not self.ready or not isinstance(self.controller,PositionController):
            self.position_enable.set(False);self.append_log('Connect before enabling position control');return
        self.position_starting=True
        if not self.post_settings():return
        self.position_ui_requested=True
        self.position_ui_token=self.controller.enable_position()
        self.position_state.set('Reading feedback / arming; targets initialize to measured positions')

    def position_stop(self):
        self.position_ui_requested=False;self.position_starting=False
        if hasattr(self,'position_enable'):self.position_enable.set(False)
        if isinstance(self.controller,PositionController):self.controller.disable_position()
        if hasattr(self,'position_state'):self.position_state.set('OFF — ZERO / DISARM requested')

    def position_focus(self):
        if self.position_ui_requested and self.root.focus_displayof() is None:self.position_stop()

    def tab_changed(self,event=None):
        if self.position_tab is not None:
            if self.main_tabs.select()!=self.position_tab and self.position_ui_requested:self.position_stop()
            elif self.main_tabs.select()==self.position_tab and self.ready:self.panic()
        super().tab_changed(event)

    def panic(self):
        self.position_stop()
        super().panic()

    def render(self,*args):
        super().render(*args)
        if not hasattr(self,'position_readouts'):return
        for i in range(3):
            config=args[2][i+1]
            flags=args[0]['motor_status'][i]
            boundary='lower and upper' if flags & 12 == 12 else 'lower' if flags & 4 else 'upper' if flags & 8 else 'neither'
            units='Normalized (0-65535)' if config[13] else 'Native (encoder counts)' if config[1] else 'Native (filtered ADC counts)'
            self.board_config_readouts[i].set(
                f"Travel limits: {'Enabled' if config[11] else 'Disabled'}    Units: {units}    Boundary: {boundary}\n"
                f"Lower: {config[5]}    Upper: {config[6]}    Output reversed: {'Yes' if config[12] else 'No'}")
        if self.position_ui_requested and not self.position_starting:return
        for i in range(3):
            try:value=str(normalized_position(args[1][i+1],args[2][i+1]))
            except ValueError:value='invalid calibration'
            self.position_readouts[i].set(f'Position: {value} / 65535')
            self.current_readouts[i].set(f'Measured current: {args[1][i+1][8]} mA')

    def pump(self):
        try:
            for _ in range(20):
                try:event=self.position_initial.get_nowait()
                except queue.Empty:event=self.position_events.get_nowait()
                if event[0]=='position_off':
                    if len(event)>1 and event[1]!=self.position_ui_token:continue
                    self.position_ui_requested=False;self.position_starting=False
                    if hasattr(self,'position_enable'):self.position_enable.set(False)
                    if hasattr(self,'position_state'):self.position_state.set('Position loop OFF — see command log')
                    for variable in getattr(self,'output_readouts',[]):variable.set('Last drive: 0')
                elif self.position_ui_requested and event[1]==self.position_ui_token:
                    _,token,initial,positions,currents,commands=event
                    if initial:
                        for i,value in enumerate(positions):
                            self.position_targets[i].set(value);self.target_text[i].set(str(value))
                        self.position_starting=False
                    for i in range(3):
                        self.position_readouts[i].set(f'Position: {positions[i]} / 65535')
                        self.current_readouts[i].set(f'Measured current: {currents[i]} mA')
                        self.output_readouts[i].set(f'Last drive: {commands[i]}')
                    self.position_state.set('Position loop enabled — move targets to command motion')
        except queue.Empty:pass
        if self.position_ui_requested and self.controller and (self.controller.stop.is_set() or not self.controller.thread.is_alive()):self.position_stop()
        super().pump()

    def close(self):
        self.position_stop()
        super().close()

    def toggle(self):
        if self.upload_thread and self.upload_thread.is_alive():
            self.append_log('Wait for the firmware upload to finish')
            return
        if self.thread and self.thread.is_alive():
            self.position_stop()
            self.jog_release()
            self.ready = False
            self.controller.halt()
            self.controller.stop.set()
            self.connection.set('Disconnecting; DISARM requested')
            self.button.configure(state='disabled')
            return
        try:
            interval = int(self.interval.get())
            if not 100 <= interval <= 5000:
                raise ValueError()
        except ValueError:
            self.append_log('Polling interval must be 100..5000 ms')
            return
        self.ready = False
        if any(self.staged):
            self.append_log('New connection: staged edits discarded; import/stage after connecting')
        self.staged = [{} for _ in range(4)]
        self.latest_config = None
        for variable in self.motor_vars:
            variable.set('0')
        for widget in self.connection_controls:
            widget.configure(state='disabled')
        if self.transport.get() == "CANable2":
            self.adapter.set(self.port.get().strip())
        self.stop.clear()
        self.repeat.set(False)
        self.last_sample = None
        self.controller = PositionController(self.port.get().strip(), int(self.node.get()), self.transport.get() == 'USB CDC', interval, self.publish)
        self.thread = self.controller.thread
        self.thread.start()
        self.button.configure(text='Disconnect', state='normal')
        self.connection.set('Connecting')



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--node',type=int,choices=range(8),default=0)
    parser.add_argument('--usb',action='store_true')
    args=parser.parse_args()
    root=tk.Tk();PositionGUI(root,args);root.mainloop()

if __name__=='__main__':main()
