"""Krabby control console v2: hold-to-run jog. Existing GUI files unchanged."""
import argparse
import ctypes
import threading
import tkinter as tk
from tkinter import ttk

from krabby import ARM, DISARM, MOTOR
from krabby_control_gui import ControlGUI, Controller, ControlNode, repeat_period


class JogNode(ControlNode):
    def request(self, op, *args, **kwargs):
        if op == DISARM:
            # A release must transmit a zero snapshot, even if the firmware has
            # already disarmed. Then explicitly remove permission to drive.
            try:
                super().request(MOTOR, data=b'\0\0\0', allow_error=True)
            except Exception:
                pass  # Still attempt DISARM when zero acknowledgment is lost.
        return super().request(op, *args, **kwargs)


class JogController(Controller):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('factory', JogNode)
        super().__init__(*args, **kwargs)
        self.hold_lock = threading.Lock()
        self.held = None
        self.hold_sequence = 0
        self.active_jog = None

    def is_held(self, token):
        with self.hold_lock:
            return self.held == token

    def press(self, motor, direction, drive):
        if motor not in range(3) or direction not in (-1, 1) or not -127 <= drive <= 127:
            raise ValueError('Motor 1..3, direction +/- and drive -127..127 required')
        with self.hold_lock:
            if self.held is not None:
                return None  # One physical jog button at a time.
            self.hold_sequence += 1
            token = self.hold_sequence
            self.held = token
        values = [0, 0, 0]
        values[motor] = direction * abs(drive)
        # Stop any previous motor-tab repetition and discard pending actions.
        # The urgent DISARM runs before the new jog job.
        self.halt()
        try:
            self.submit('jog', token, tuple(values))
        except Exception:
            self.release()
            raise
        return token

    def release(self):
        with self.hold_lock:
            self.held = None
        self.halt()  # Cancellation is visible even while ARM is awaiting a reply.

    def disarm(self, node):
        if self.active_jog is not None:
            with self.hold_lock:
                if self.held == self.active_jog:
                    self.held = None
        self.active_jog = None
        super().disarm(node)

    def action(self, node, name, args):
        if name != 'jog':
            with self.hold_lock:
                self.held = None
            if self.active_jog is not None:
                self.disarm(node)
            return super().action(node, name, args)
        token, values = args
        try:
            if not self.is_held(token) or self.stop.is_set():
                return
            super().disarm(node)
            self.watchdog = node.page(0)[1]
            self.period = repeat_period(20, self.watchdog)
            if not self.is_held(token) or self.stop.is_set():
                return
            node.request(ARM)
            self.generation += 1
            self.arm_permitted = True
            self.values = (0, 0, 0)
            if not self.is_held(token) or self.stop.is_set() or self.urgent.is_set():
                self.disarm(node)
                return
            self.active_jog = token
            self.values = values
            self.last_tx = None
            self.max_gap = 0
            self.repeating = True
            self.drive(node)
            if self.active_jog is not None:
                self.publish(('drive', True, values, f'Jog held: {values}; interval {self.period*1000:g} ms'))
        except Exception:
            with self.hold_lock:
                if self.held == token:
                    self.held = None
            raise

    def drive(self, node):
        token = self.active_jog
        if token is not None and not self.is_held(token):
            self.disarm(node)
            return
        super().drive(node)
        # Release may arrive while a serial transaction was in progress. Never
        # leave the next nonzero refresh waiting for its ordinary deadline.
        if token is not None and (not self.is_held(token) or self.stop.is_set()):
            self.disarm(node)


class JogGUI(ControlGUI):
    def __init__(self, root, args):
        self.jog_token = None
        self.jog_button = None
        super().__init__(root, args)
        root.title('Krabby control console v2 — hold-to-run jog')
        frame = ttk.Frame(self.main_tabs, padding=20)
        self.main_tabs.add(frame, text='Jog')
        self.jog_tab = str(frame)
        self.jog_drives = [tk.StringVar(value='20') for _ in range(3)]
        self.jog_currents = [tk.StringVar(value='Measured: -- mA') for _ in range(3)]
        ttk.Label(frame, text='Hold a direction button to drive. Release sends ZERO for all motors, then DISARM.', font=('Segoe UI', 12)).grid(row=0, column=0, columnspan=4, sticky='w', pady=(0,20))
        for row, variable in enumerate(self.jog_drives, 1):
            ttk.Label(frame, text=f'Motor {row}').grid(row=row, column=0, padx=10, pady=18)
            minus = ttk.Button(frame, text='◀  Negative', takefocus=False, width=20)
            minus.grid(row=row, column=1, padx=10)
            ttk.Spinbox(frame, from_=-127, to=127, textvariable=variable, width=10).grid(row=row, column=2, padx=10)
            plus = ttk.Button(frame, text='Positive  ▶', takefocus=False, width=20)
            plus.grid(row=row, column=3, padx=10)
            ttk.Label(frame, textvariable=self.jog_currents[row-1], width=24).grid(row=row, column=4, padx=15, sticky='w')
            for button, direction in ((minus,-1),(plus,1)):
                button.bind('<ButtonPress-1>', lambda event,m=row-1,d=direction: self.jog_press(event,m,d))
                button.bind('<ButtonRelease-1>', self.jog_release)
        ttk.Label(frame, text='Drive uses the absolute magnitude of the field; the button supplies its sign.\nDefault magnitude 20. Only one motor/button can jog at once; other motors receive zero.\nPress automatically stops existing repetition, ARMs, then sends fresh commands.\nValues are latched on press; release and press again to change drive.').grid(row=4, column=0, columnspan=4, sticky='w', pady=20)
        self.jog_status = tk.StringVar(value='Jog idle — no button held')
        ttk.Label(frame, textvariable=self.jog_status).grid(row=5, column=0, columnspan=4, sticky='w')
        ttk.Button(frame, text='STOP / DISARM', command=self.panic).grid(row=6, column=0, columnspan=4, sticky='w', pady=20)
        ttk.Label(frame, text='Release outside the button, window focus loss, minimization, tab changes and Escape cancel jogging.\nThe board watchdog remains essential if the PC or connection fails; do not disable it for motor testing.').grid(row=7, column=0, columnspan=4, sticky='w')
        root.bind_all('<ButtonRelease-1>', self.jog_release, add='+')
        root.bind('<FocusOut>', lambda _e: root.after(30, self.check_focus), add='+')
        root.bind('<Unmap>', lambda e: self.jog_release() if e.widget == root else None, add='+')
        root.bind('<Escape>', lambda _e: self.panic(), add='+')
        self.main_tabs.bind('<<NotebookTabChanged>>', self.tab_changed, add='+')
        root.after(30, self.check_mouse)

    def render(self, *args):
        super().render(*args)
        sensors = args[1]
        for motor, variable in enumerate(self.jog_currents, 1):
            variable.set(f'Measured: {sensors[motor][8]} mA')

    def toggle(self):
        if self.upload_thread and self.upload_thread.is_alive():
            self.append_log('Wait for the firmware upload to finish')
            return
        if self.thread and self.thread.is_alive():
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
        self.controller = JogController(self.port.get().strip(), int(self.node.get()), self.transport.get() == 'USB CDC', interval, self.publish)
        self.thread = self.controller.thread
        self.thread.start()
        self.button.configure(text='Disconnect', state='normal')
        self.connection.set('Connecting')

    def jog_press(self, event, motor, direction):
        if self.jog_token is not None:
            return 'break'
        if not self.ready or not isinstance(self.controller, JogController):
            self.append_log('Connect before jogging')
            return 'break'
        try:
            drive = int(self.jog_drives[motor].get())
            if not -127 <= drive <= 127:
                raise ValueError('Drive must be -127..127')
            token = self.controller.press(motor,direction,drive)
            if token is None:
                return 'break'
            self.jog_token = token
            self.jog_button = event.widget
            event.widget.grab_set()
            event.widget.state(['pressed'])
            self.jog_status.set(f'Held: M{motor+1}, command {direction*abs(drive)}; arming/refreshing')
        except Exception as error:
            self.jog_release()
            self.append_log('Jog: ' + str(error))
        return 'break'

    def jog_release(self, _event=None):
        if self.jog_token is None:
            return
        self.jog_token = None
        button, self.jog_button = self.jog_button, None
        if isinstance(self.controller, JogController):
            self.controller.release()
        if button is not None:
            try:
                button.state(['!pressed'])
                button.grab_release()
            except tk.TclError:
                pass
        self.jog_status.set('Released — ZERO / DISARM requested')

    def check_focus(self):
        if self.jog_token is not None and self.root.focus_displayof() is None:
            self.jog_release()

    def check_mouse(self):
        if self.jog_token is not None:
            if self.thread is None or not self.thread.is_alive() or not self.ready:
                self.jog_release()
            elif not self.controller.is_held(self.jog_token):
                self.jog_release()
            elif sys_platform_windows() and not ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000:
                # Fallback for a release outside Tk's mouse capture.
                self.jog_release()
        if not self.closing:
            self.root.after(30,self.check_mouse)

    def tab_changed(self, _event=None):
        if self.main_tabs.select() == self.jog_tab and self.ready:
            # Entering hold-to-run mode must not leave an old motor-tab stream active.
            self.panic()
        else:
            self.jog_release()

    def panic(self):
        self.jog_release()
        super().panic()

    def close(self):
        self.jog_release()
        super().close()


def sys_platform_windows():
    import sys
    return sys.platform == 'win32'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--node',type=int,choices=range(8),default=0)
    parser.add_argument('--usb',action='store_true')
    args=parser.parse_args()
    root=tk.Tk();JogGUI(root,args);root.mainloop()

if __name__=='__main__':main()
