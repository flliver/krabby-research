"""Writable Krabby commissioning console. The read-only viewer is unchanged."""
import argparse
from collections import deque
import json
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog

from krabby import (Node, VERSION, STATUS, SENSORS, GET, SET, DEFAULTS, SAVE, LOAD,
                    ARM, DISARM, MOTOR, REBOOT_ARM, REBOOT_NOW)
from krabby_debug_gui import DebugGUI, ROOT
from openblt_canable import LENGTHS


class ControlNode(Node):
    """Capture CAN telemetry while preserving single-request serial ownership."""
    def __init__(self, *args, telemetry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = telemetry or (lambda data: None)
        if not self.usb:
            self.link.port.timeout = .001

    def drain(self):
        if self.usb:
            return []
        port = self.link.port
        self.link.pending.extend(port.read(port.in_waiting) if port.in_waiting else b'')
        frames = []
        while b'\r' in self.link.pending:
            line, _, rest = self.link.pending.partition(b'\r')
            self.link.pending[:] = rest
            line = line.strip()
            if not line or line[:1] not in (b'b', b'd', b't', b'B', b'D', b'T'):
                continue
            width = 8 if line[:1].isupper() else 3
            identifier = int(line[1:1+width], 16)
            dlc = int(line[1+width:2+width], 16)
            data = bytes.fromhex(line[2+width:].decode())
            if len(data) != LENGTHS[dlc]:
                raise RuntimeError('CAN record length mismatch')
            if identifier == 0x600 + self.node:
                self.telemetry(data)
            else:
                frames.append((identifier, data))
        if len(self.link.pending) > 512:
            raise RuntimeError('Adapter record overflow')
        return frames

    def raw(self, packet, timeout=1):
        if self.usb:
            return super().raw(packet, timeout)
        self.link.send(packet, 0x500 + self.node)
        deadline = time.monotonic() + timeout
        sequence = packet[1] if len(packet) > 1 else 0
        while time.monotonic() < deadline:
            for identifier, data in self.drain():
                if identifier == 0x580 + self.node and len(data) >= 5 and data[1] == sequence:
                    return data
            time.sleep(.0005)
        raise TimeoutError('No application response')


def repeat_period(requested_ms, watchdog_units):
    if not 1 <= requested_ms <= 1000:
        raise ValueError('Repeat interval must be 1..1000 ms')
    return min(requested_ms / 1000, watchdog_units * .01 / 4) if watchdog_units else requested_ms / 1000


class Controller:
    """One worker owns all I/O; motor deadlines precede monitoring and queued jobs."""
    def __init__(self, port, node, usb, poll_ms, publish, factory=ControlNode):
        self.args = port, node, usb
        self.poll = poll_ms / 1000
        self.publish = publish
        self.factory = factory
        self.jobs = queue.Queue(maxsize=24)
        self.stop = threading.Event()
        self.urgent = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.repeating = False
        self.arm_permitted = False
        self.generation = 0
        self.values = (0, 0, 0)
        self.last_tx = None
        self.max_gap = 0
        self.period = .02
        self.watchdog = 0
        self.key = None
        self.key_until = 0
        self.rebooting = False
        self.telemetry_latest = {}
        self.telemetry_count = 0
        self.last_telemetry = 0

    def submit(self, name, *args):
        if name == 'motor':
            args = (*args, self.generation if self.arm_permitted else None)
        self.jobs.put_nowait((name, args))

    def halt(self):
        self.urgent.set()
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break

    def message(self, text):
        self.publish(('message', text))

    def telemetry(self, data):
        self.telemetry_count += 1
        if len(data) >= 5 and data[0] == 1:
            self.telemetry_latest[(data[2], data[1])] = bytes(data)
        now = time.monotonic()
        if now - self.last_telemetry >= .1:
            self.last_telemetry = now
            self.publish(('telemetry', self.telemetry_count, dict(self.telemetry_latest)))

    def disarm(self, node):
        self.arm_permitted = False
        self.generation += 1
        self.repeating = False
        self.values = (0, 0, 0)
        self.publish(('drive', False, self.values, 'Stopped'))
        node.request(DISARM)

    def drive(self, node):
        if self.stop.is_set() or self.urgent.is_set():
            return
        now = time.monotonic()
        if self.last_tx is not None and self.watchdog and now - self.last_tx >= self.watchdog * .01 * .75:
            raise RuntimeError('Motor deadline missed; stopped. Explicit ARM required.')
        node.motor(*self.values)
        if self.last_tx is not None:
            self.max_gap = max(self.max_gap, now - self.last_tx)
        self.last_tx = now
        self.next_motor = now + self.period

    def action(self, node, name, args):
        # Mutating configuration never runs behind an active motor stream.
        if name in ('motor_config', 'field', 'page', 'defaults', 'save', 'load', 'reboot_arm', 'reboot_now', 'reboot', 'raw', 'usb_line'):
            self.disarm(node)
        if name in ('arm', 'arm_repeat'):
            if name == 'arm_repeat':
                repeat_period(args[0], 0)  # Validate before sending ARM.
            self.repeating = False
            self.values = (0, 0, 0)
            self.watchdog = node.page(0)[1]
            node.request(ARM)
            self.generation += 1
            self.arm_permitted = True
            self.publish(('drive', False, self.values, 'ARM acknowledged; commands cleared'))
            if name == 'arm_repeat' and not self.stop.is_set() and not self.urgent.is_set():
                self.start_repeat(node, args[0], prepared=True)
        elif name == 'disarm':
            self.disarm(node)
        elif name == 'motor':
            ticket = args[1] if len(args) > 1 else self.generation
            if not self.arm_permitted or ticket != self.generation:
                raise RuntimeError('Motor request predates ARM acknowledgment; apply fresh values')
            values = tuple(args[0])
            if len(values) != 3 or any(not -127 <= v <= 127 for v in values):
                raise ValueError('Motor values must be -127..127')
            sent = time.monotonic()
            node.motor(*values)
            self.values = values
            if self.last_tx is not None:
                self.max_gap = max(self.max_gap, sent - self.last_tx)
            self.last_tx = sent
            self.next_motor = self.last_tx + self.period
        elif name == 'repeat':
            self.start_repeat(node, args[0])
        elif name == 'motor_config':
            motor, reverse, normalized, lower, upper = args
            if motor not in range(3) or not -2147483648 <= lower < upper <= 2147483647 or (normalized and not 0 <= lower < upper <= 65535):
                raise ValueError('Invalid motor or travel limits')
            self.disarm(node)
            values = list(node.page(motor+1))
            values[5:7] = [lower, upper]
            values[12:14] = [int(reverse), int(normalized)]
            node.write_page(motor+1, values)
            self.message('Motor polarity/limits written to RAM; SAVE to persist')
        elif name == 'field':
            node.set(*args)
            self.publish(('clear_edits', args[0], [args[1]]))
        elif name == 'page':
            page, edits = args
            values = list(node.page(page))
            for field, value in edits.items():
                values[int(field)] = int(value)
            node.write_page(page, values)
            self.publish(('clear_edits', page, list(edits)))
        elif name == 'defaults':
            node.request(DEFAULTS, args[0])
            self.publish(('clear_all', args[0]))
        elif name == 'save':
            node.request(SAVE)
        elif name == 'load':
            node.request(LOAD)
            self.publish(('clear_all', 255))
        elif name == 'version':
            self.message('VERSION ' + repr(struct.unpack('<4I', node.request(VERSION))))
        elif name == 'refresh':
            pass
        elif name == 'reboot_arm':
            self.key = node.request(REBOOT_ARM)
            self.key_until = time.monotonic() + .5
            self.message('Reboot key ' + self.key.hex() + '; expires in 500 ms')
        elif name == 'reboot_now':
            if self.key is None or time.monotonic() >= self.key_until:
                raise ValueError('Reboot permission expired; use combined two-stage reboot or arm again')
            key, self.key = self.key, None
            node.request(REBOOT_NOW, data=key)
            self.rebooting = True
            self.stop.set()
        elif name == 'reboot':
            node.reboot()
            self.rebooting = True
            self.stop.set()
        elif name == 'raw':
            packet = bytes.fromhex(args[0])
            if len(packet) > 64:
                raise ValueError('Maximum 64 bytes')
            self.message('RAW response ' + node.raw(packet).hex(' ').upper())
        elif name == 'usb_line':
            if not node.usb:
                raise ValueError('Legacy text commands require a USB CDC connection')
            line = args[0]
            if '\r' in line or '\n' in line:
                raise ValueError('One line at a time')
            node.link.reset_input_buffer()
            node.link.write(line.encode('ascii') + b'\n')
            deadline = time.monotonic() + .25
            response = bytearray()
            while time.monotonic() < deadline:
                response.extend(node.link.read(node.link.in_waiting or 1))
            self.message('USB ' + response.decode(errors='replace').strip())
        else:
            raise ValueError('Unknown action')
        self.message(name.upper() + ' acknowledged')
        self.config_due = 0

    def start_repeat(self, node, requested_ms, prepared=False):
        if not self.arm_permitted:
            raise RuntimeError('Use this console to ARM before starting repeat')
        if not prepared:
            status = node.status()
            if not status['node_status'] & 8:
                raise RuntimeError('Board is disarmed. Use ARM + repeat zero, then apply new values.')
            self.watchdog = node.page(0)[1]
        self.period = repeat_period(requested_ms, self.watchdog)
        self.last_tx = None
        self.max_gap = 0
        self.repeating = True
        self.drive(node)
        label = f'Repeating every {self.period*1000:g} ms'
        if not self.watchdog:
            label += ' — board watchdog DISABLED'
        self.publish(('drive', True, None, label))

    def run(self):
        node = None
        confirmed = False
        try:
            port, number, usb = self.args
            node = self.factory(port, number, usb, telemetry=self.telemetry)
            version = struct.unpack('<4I', node.request(VERSION))
            if version[:2] != (1, 4):
                raise RuntimeError(f'Unsupported protocol/schema {version[:2]}')
            confirmed = True
            config = [node.page(page) for page in range(4)]
            sensors = [node.sensors(page) for page in range(4)]
            status = node.status()
            uid = node.uid() if version[3] & 32 else 'Unavailable (older firmware)'
            self.publish(('connected', uid))
            self.publish(('sample', status, sensors, config, time.monotonic(), time.monotonic()))
            self.config_due = time.monotonic() + 2
            config_time = time.monotonic()
            next_read = time.monotonic()
            read_index = 0
            config_index = None
            while not self.stop.is_set():
                if self.urgent.is_set():
                    self.disarm(node)
                    self.urgent.clear()
                    self.message('STOP / DISARM acknowledged')
                now = time.monotonic()
                if self.repeating and now >= self.next_motor:
                    try:
                        self.drive(node)
                    except Exception as error:
                        self.disarm(node)
                        self.message(str(error))
                    continue
                try:
                    name, args = self.jobs.get_nowait()
                except queue.Empty:
                    name = None
                if name is not None:
                    if self.urgent.is_set():
                        continue
                    try:
                        self.action(node, name, args)
                    except (RuntimeError, ValueError, struct.error) as error:
                        self.disarm(node)
                        self.message(f'{name.upper()} failed: {error}; no automatic retry')
                    continue
                if now >= next_read:
                    # One RPC per scheduling turn; motor deadlines checked between reads.
                    if read_index == 0:
                        status = node.status()
                        if self.repeating and not status['node_status'] & 8:
                            self.disarm(node)
                            self.message('Board disarmed; repeat stopped. Explicit ARM required.')
                    else:
                        sensors[read_index-1] = node.sensors(read_index-1)
                    read_index = (read_index + 1) % 5
                    next_read = now + self.poll / 5
                    self.publish(('sample', dict(status), list(sensors), list(config), time.monotonic(), config_time))
                    if self.repeating:
                        self.publish(('timing', self.max_gap * 1000))
                    continue
                if config_index is None and now >= self.config_due:
                    config_index = 0
                if config_index is not None:
                    config[config_index] = node.page(config_index)
                    config_index += 1
                    if config_index == 4:
                        config_index = None
                        config_time = time.monotonic()
                        self.config_due = config_time + 2
                    continue
                if not usb:
                    node.drain()
                wait = min(.005, max(0, next_read-time.monotonic()))
                if self.repeating:
                    wait = min(wait, max(0, self.next_motor-time.monotonic()))
                # Python sleep uses the Windows high-resolution waitable timer;
                # Event.wait can quantize short waits to the system timer tick.
                # Keep sleeps <=1 ms so STOP remains responsive.
                time.sleep(min(wait, .001))
        except Exception as error:
            self.publish(('error', str(error)))
        finally:
            self.repeating = False
            if node:
                try:
                    if confirmed and not self.rebooting:
                        node.request(DISARM)
                        self.message('Disconnect: DISARM acknowledged')
                except Exception as error:
                    self.message('Disconnect: DISARM unconfirmed: ' + str(error))
                try:
                    node.close()
                except Exception as error:
                    self.message('Port close: ' + str(error))
            self.publish(('drive', False, (0, 0, 0), 'Disconnected; repeat OFF'))


class ControlGUI(DebugGUI):
    def __init__(self, root, args):
        self.controller = None
        self.upload_thread = None
        self.staged = [{} for _ in range(4)]
        self.latest_config = None
        self.ready = False
        super().__init__(root, args)
        self.events = queue.Queue(maxsize=256)
        root.title('Krabby control console — writable commissioning tool')
        root.geometry('1240x900')
        notebook = next(w for w in root.winfo_children() if isinstance(w, ttk.Notebook))
        self.main_tabs = notebook
        bar = root.winfo_children()[0]
        self.connection_controls = [w for w in bar.winfo_children() if isinstance(w, (ttk.Combobox, ttk.Spinbox))]
        config = notebook.winfo_children()[1]
        self.pages = next(w for w in config.winfo_children() if isinstance(w, ttk.Notebook))
        tools = ttk.Frame(config)
        tools.pack(fill='x', before=self.pages)
        for text, command in (
            ('Stage selected field', self.edit_field), ('Write selected field to RAM', self.write_field),
            ('Write staged page to RAM', self.write_page), ('Discard edits', self.discard),
            ('Refresh', lambda: self.issue('refresh')),
        ):
            ttk.Button(tools, text=text, command=command).pack(side='left', padx=3, pady=4)
        tools2 = ttk.Frame(config)
        tools2.pack(fill='x', before=self.pages)
        for text, command in (
            ('Defaults: selected page', lambda: self.issue('defaults', self.page_index())),
            ('Defaults: all pages', lambda: self.issue('defaults', 255)),
            ('LOAD flash → RAM', lambda: self.issue('load')),
            ('SAVE active RAM → flash', lambda: self.issue('save')),
            ('Export RAM JSON', self.export_config), ('Import JSON as staged edits', self.import_config),
        ):
            ttk.Button(tools2, text=text, command=command).pack(side='left', padx=3, pady=4)
        ttk.Label(config, text='* = staged edit, not on board. Write RAM before SAVE. Configuration operations stop repeat and DISARM.').pack(anchor='w', before=self.pages)
        for table in self.config_tables:
            table.column('Range', width=230)
            table.bind('<Double-1>', lambda _event: self.edit_field())
        motor = ttk.Frame(notebook, padding=12)
        notebook.add(motor, text='Motor control')
        self.motor_vars = [tk.StringVar(value='0') for _ in range(3)]
        for i, variable in enumerate(self.motor_vars):
            ttk.Label(motor, text=f'M{i+1} signed duty (-127..127)').grid(row=i, column=0, sticky='w', pady=8)
            ttk.Spinbox(motor, from_=-127, to=127, textvariable=variable, width=10).grid(row=i, column=1, padx=12)
        ttk.Label(motor, text='Edits are latched only by Apply. ARM clears all three values.').grid(row=3, column=0, columnspan=4, sticky='w')
        buttons = ttk.Frame(motor)
        buttons.grid(row=4, column=0, columnspan=5, sticky='w', pady=12)
        for text, command in (
            ('ARM (neutral only)', lambda: self.issue('arm')),
            ('ARM + repeat ZERO', self.arm_repeat),
            ('Apply / send three motor values', self.apply_motor),
            ('ZERO commands', self.zero_motor),
            ('STOP repeat / DISARM', self.panic),
        ):
            ttk.Button(buttons, text=text, command=command).pack(side='left', padx=3)
        self.repeat = tk.BooleanVar(value=False)
        self.repeat_ms = tk.StringVar(value='20')
        ttk.Checkbutton(motor, text='Repeat latched motor values', variable=self.repeat, command=self.toggle_repeat).grid(row=5, column=0, sticky='w')
        ttk.Spinbox(motor, from_=1, to=1000, textvariable=self.repeat_ms, width=8).grid(row=5, column=1)
        ttk.Label(motor, text='Requested interval (ms); automatically capped at watchdog / 4').grid(row=5, column=2, sticky='w')
        self.drive_status = tk.StringVar(value='Repeat OFF')
        self.gap = tk.StringVar(value='Maximum observed motor TX gap: --')
        ttk.Label(motor, textvariable=self.drive_status).grid(row=6, column=0, columnspan=5, sticky='w', pady=8)
        ttk.Label(motor, textvariable=self.gap).grid(row=7, column=0, columnspan=5, sticky='w')
        ttk.Label(motor, text='Windows timing is best effort. Deadline misses stop the stream and DISARM; never automatically re-arm.\nStopping/disconnecting applies configured brake or Hi-Z. With watchdog=0, host failure is not bounded by a command timeout.').grid(row=8, column=0, columnspan=5, sticky='w', pady=12)
        advanced = ttk.Frame(notebook, padding=10)
        notebook.add(advanced, text='Management / raw commands')
        management = ttk.Frame(advanced)
        management.pack(fill='x')
        for text, name in (('Protocol / capabilities', 'version'), ('REBOOT_ARM (500 ms)', 'reboot_arm'), ('REBOOT_NOW', 'reboot_now'), ('Reboot (both stages)', 'reboot')):
            ttk.Button(management, text=text, command=lambda n=name: self.issue(n)).pack(side='left', padx=3)
        self.raw_packet = tk.StringVar(value='01 01 02 00 00 00')
        ttk.Label(advanced, text='Raw application request bytes (version, sequence, opcode, argument, offset, count, payload):').pack(anchor='w', pady=(14, 3))
        ttk.Entry(advanced, textvariable=self.raw_packet, width=95).pack(fill='x')
        ttk.Button(advanced, text='Send raw request (stops repeat / disarms first)', command=lambda: self.issue('raw', self.raw_packet.get())).pack(anchor='w', pady=4)
        self.usb_line = tk.StringVar(value='can status')
        ttk.Label(advanced, text='USB legacy text commands: hello, status, can status, led on/off, tx/txfd/... (USB connection only)').pack(anchor='w', pady=(12, 3))
        ttk.Entry(advanced, textvariable=self.usb_line, width=95).pack(fill='x')
        ttk.Button(advanced, text='Send USB line', command=lambda: self.issue('usb_line', self.usb_line.get())).pack(anchor='w', pady=4)
        self.telemetry_text = tk.StringVar(value='CAN telemetry: none received. Configure node telemetry_ms to enable.')
        ttk.Label(advanced, textvariable=self.telemetry_text, wraplength=1100).pack(anchor='w', pady=12)
        self.telemetry_rows = self.table(advanced, ('Topic', 'Latest decoded payload'),
                                        [(name, '--') for name in ('Status', 'Board', 'M1', 'M2', 'M3')], height=5)
        self.telemetry_rows.column('Latest decoded payload', width=850)
        update = ttk.Frame(notebook, padding=12)
        notebook.add(update, text='Firmware update')
        self.image = tk.StringVar()
        self.adapter = tk.StringVar(value='COM7')
        ttk.Entry(update, textvariable=self.image, width=95).pack(fill='x', pady=6)
        ttk.Button(update, text='Choose application SREC / BIN', command=self.choose_image).pack(anchor='w')
        ttk.Label(update, text='CANable port for OpenBLT (also used when connected by USB):').pack(anchor='w', pady=(15, 3))
        ttk.Entry(update, textvariable=self.adapter).pack(anchor='w')
        ttk.Button(update, text='DISARM, disconnect, upload and verify', command=self.upload).pack(anchor='w', pady=15)
        ttk.Label(update, text='Uses the existing OpenBLT uploader. Application only; configuration retained.\nThe console stays disconnected after upload. Reconnect after USB re-enumerates.').pack(anchor='w')
        self.log = tk.Text(root, height=7, wrap='word', state='disabled')
        self.log.pack(fill='x', padx=8, pady=5)
        self.append_log('Writable console. Staged edits are separate from active RAM and saved flash. Repeat starts OFF.')

    def append_log(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', time.strftime('%H:%M:%S ') + text + '\n')
        if int(self.log.index('end-1c').split('.')[0]) > 400:
            self.log.delete('1.0', '101.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def page_index(self):
        return self.pages.index(self.pages.select())

    def issue(self, name, *args):
        if not self.ready or not self.controller or not self.thread or not self.thread.is_alive():
            self.append_log('Connect before sending commands')
            self.repeat.set(False)
            return
        if name in ('arm', 'arm_repeat', 'disarm'):
            for variable in self.motor_vars:
                variable.set('0')
        try:
            self.controller.submit(name, *args)
        except queue.Full:
            self.append_log('Command queue full; command not sent')

    def panic(self):
        self.repeat.set(False)
        for variable in self.motor_vars:
            variable.set('0')
        if self.controller:
            self.controller.halt()
        self.drive_status.set('STOP requested; awaiting DISARM acknowledgment')

    def toggle_repeat(self):
        if self.repeat.get():
            try:
                self.issue('repeat', int(self.repeat_ms.get()))
            except ValueError:
                self.repeat.set(False)
                self.append_log('Repeat interval must be an integer')
        else:
            self.panic()

    def arm_repeat(self):
        try:
            self.issue('arm_repeat', int(self.repeat_ms.get()))
        except ValueError:
            self.append_log('Repeat interval must be an integer')

    def apply_motor(self):
        try:
            values = tuple(int(v.get()) for v in self.motor_vars)
            if any(not -127 <= v <= 127 for v in values):
                raise ValueError()
            self.issue('motor', values)
        except ValueError:
            self.append_log('Each motor value must be an integer -127..127')

    def zero_motor(self):
        for variable in self.motor_vars:
            variable.set('0')
        self.apply_motor()

    def edit_field(self):
        page = self.page_index()
        selection = self.config_tables[page].selection()
        if not selection:
            self.append_log('Select a configuration row first')
            return
        field = int(selection[0])
        fields = self.schema['node' if page == 0 else 'motors']
        metadata = next((f for f in fields if f['id'] == field), dict(name='reserved', minimum=0, maximum=0))
        current = self.staged[page].get(field, self.latest_config[page][field] if self.latest_config else 0)
        value = simpledialog.askinteger('Stage configuration value', f"Page {page}, field {field}: {metadata['name']}\n{metadata['minimum']} .. {metadata['maximum']}", initialvalue=current, minvalue=metadata['minimum'], maxvalue=metadata['maximum'], parent=self.root)
        if value is not None:
            self.staged[page][field] = value
            self.paint_edits()

    def paint_edits(self):
        for page, edits in enumerate(self.staged):
            for field, value in edits.items():
                table = self.config_tables[page]
                row = list(table.item(str(field), 'values'))
                row[3] = f'{value} *'
                table.item(str(field), values=row)

    def write_field(self):
        page = self.page_index()
        selection = self.config_tables[page].selection()
        if not selection or int(selection[0]) not in self.staged[page]:
            self.append_log('Stage a value for the selected field first')
            return
        field = int(selection[0])
        self.issue('field', page, field, self.staged[page][field])

    def write_page(self):
        page = self.page_index()
        if not self.staged[page]:
            self.append_log('No staged edits on this page')
            return
        self.issue('page', page, dict(self.staged[page]))

    def discard(self):
        self.staged[self.page_index()].clear()
        self.issue('refresh')

    def export_config(self):
        if self.latest_config is None:
            self.append_log('Read configuration before exporting')
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            Path(path).write_text(json.dumps(dict(schema=4, pages=self.latest_config), indent=2)+'\n')
            self.append_log('Exported active RAM (staged edits excluded): ' + path)

    def import_config(self):
        path = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            pages = data['pages']
            from krabby import migrate_pages
            pages = migrate_pages(data['schema'], pages)
            if any(type(v) is not int or not -2147483648 <= v <= 2147483647 for p in pages for v in p):
                raise ValueError('Values must be signed 32-bit integers')
            self.staged = [dict(enumerate(p)) for p in pages]
            self.paint_edits()
            self.append_log('Imported as staged edits only. Write each page to RAM, then SAVE if wanted.')
        except Exception as error:
            self.append_log('Import failed: ' + str(error))

    def render(self, *args):
        super().render(*args)
        self.actual_node = args[0]["node"]
        self.latest_config = [list(p) for p in args[2]]
        self.paint_edits()

    def toggle(self):
        if self.upload_thread and self.upload_thread.is_alive():
            self.append_log('Wait for the firmware upload to finish')
            return
        if self.thread and self.thread.is_alive():
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
        self.controller = Controller(self.port.get().strip(), int(self.node.get()), self.transport.get() == 'USB CDC', interval, self.publish)
        self.thread = self.controller.thread
        self.thread.start()
        self.button.configure(text='Disconnect', state='normal')
        self.connection.set('Connecting')

    def pump(self):
        try:
            for _ in range(200):
                event = self.events.get_nowait()
                kind = event[0]
                if kind == 'sample' and self.ready:
                    self.render(*event[1:])
                elif kind == 'connected':
                    self.mcu_uid.set(event[1])
                    self.ready = True
                elif kind == 'error':
                    self.ready = False
                    self.connection.set('Disconnected: ' + event[1])
                    self.append_log(event[1])
                elif kind == 'message':
                    self.append_log(event[1])
                elif kind == 'drive':
                    self.repeat.set(event[1])
                    if event[2] is not None:
                        for variable, value in zip(self.motor_vars, event[2]):
                            variable.set(str(value))
                    self.drive_status.set(event[3])
                elif kind == 'timing':
                    self.gap.set(f'Maximum observed motor TX gap: {event[1]:.2f} ms')
                elif kind == 'clear_edits':
                    for field in event[2]:
                        self.staged[event[1]].pop(int(field), None)
                elif kind == 'clear_all':
                    for page in range(4) if event[1] == 255 else [event[1]]:
                        self.staged[page].clear()
                elif kind == 'telemetry':
                    self.telemetry_text.set(f'CAN telemetry frames received: {event[1]} (USB connections cannot receive CAN telemetry)')
                    for (op, sequence), data in event[2].items():
                        if len(data) < 5 or data[3] or data[4] > len(data)-5:
                            continue
                        payload = data[5:5+data[4]]
                        if op == 2 and len(payload) == 24:
                            row = 0
                            text = f'node={payload[4]}, flags=0x{payload[5]:02X}, motors={list(payload[6:9])}, commands={struct.unpack_from("<3b",payload,9)}'
                        elif op == 3 and len(payload) == 36 and 1 <= sequence <= 4:
                            row = sequence
                            values = struct.unpack('<8Ii' if row == 1 else '<9i', payload)
                            text = repr(values)
                        else:
                            continue
                        prior = self.telemetry_rows.item(str(row), 'values')
                        self.telemetry_rows.item(str(row), values=(prior[0], text))
        except queue.Empty:
            pass
        if self.thread and not self.thread.is_alive():
            self.thread = None
            self.ready = False
            self.repeat.set(False)
            self.button.configure(text='Connect', state='normal')
            self.connection.set('Disconnected; displayed values are stale')
            for widget in self.connection_controls:
                widget.configure(state='normal' if isinstance(widget, ttk.Spinbox) or str(widget.cget('textvariable')) == str(self.port) else 'readonly')
        if self.closing and self.thread is None and not (self.upload_thread and self.upload_thread.is_alive()):
            self.root.destroy()
            return
        self.root.after(25, self.pump)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[('Application', '*.srec *.s19 *.s28 *.s37 *.bin'), ('All files', '*')])
        if path:
            self.image.set(path)

    def upload(self):
        from openblt_canable import load_image
        try:
            image = Path(self.image.get()).resolve()
            load_image(image)
            port = self.adapter.get().strip()
            if not port:
                raise ValueError('Choose the CANable port')
        except Exception as error:
            self.append_log('Image validation failed: ' + str(error))
            return
        if self.upload_thread and self.upload_thread.is_alive():
            return
        actual_node = self.actual_node if self.ready and hasattr(self, "actual_node") else int(self.node.get())
        if self.controller:
            self.controller.halt()
            self.controller.stop.set()
        self.ready = False
        self.repeat.set(False)
        owner = self.thread
        def run_upload():
            if owner:
                owner.join()
            command = [sys.executable, str(ROOT/'tools/openblt_canable.py'), str(image), '--port', port, '--node', str(actual_node)]
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace',
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                for line in process.stdout:
                    self.publish(('message', line.rstrip()))
                self.publish(('message', f'Uploader exit {process.wait()}; reconnect manually'))
            except Exception as error:
                self.publish(('message', 'Uploader failed: ' + str(error)))
        self.upload_thread = threading.Thread(target=run_upload, daemon=True)
        self.upload_thread.start()
        self.append_log('Firmware update requested; waiting for DISARM/port release')

    def close(self):
        self.closing = True
        self.ready = False
        if self.controller:
            self.controller.halt()
            self.controller.stop.set()
        self.button.configure(state='disabled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--node', type=int, choices=range(8), default=0)
    parser.add_argument('--usb', action='store_true')
    args = parser.parse_args()
    root = tk.Tk()
    ControlGUI(root, args)
    root.mainloop()

if __name__ == '__main__':
    main()
