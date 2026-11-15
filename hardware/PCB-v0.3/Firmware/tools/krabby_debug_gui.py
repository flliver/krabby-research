"""Read-only Krabby debug GUI. Requires Python tkinter and pyserial."""
import argparse
import json
from pathlib import Path
import queue
import struct
import threading
import time
import tkinter as tk
from tkinter import ttk
from serial.tools import list_ports
from krabby import Node, VERSION

ROOT = Path(__file__).resolve().parents[1]
MOTOR_ROWS = [
    ('Current ADC raw', 'counts', ('PA0', 'PA1', 'PA2')),
    ('Current ADC filtered', 'counts', ('PA0', 'PA1', 'PA2')),
    ('Current sense', 'mV', ('PA0', 'PA1', 'PA2')),
    ('Position ADC raw', 'counts', ('PA7', 'PC4', 'PC5')),
    ('Position ADC filtered', 'counts', ('PA7', 'PC4', 'PC5')),
    ('Encoder', 'counts', ('TIM2', 'TIM8', 'TIM5')),
    ('Reported position', 'native / normalized', ('', '', '')),
    ('DAC register readback', 'code', ('A', 'B', 'C')),
    ('Nominal current', 'mA', ('', '', '')),
]
BOARD_ROWS = [
    ('Temperature ADC raw', 'counts', 'PC0'),
    ('Temperature ADC filtered', 'counts', 'PC0'),
    ('USB VBUS ADC raw', 'counts', 'PA6'),
    ('USB VBUS ADC filtered', 'counts', 'PA6'),
    ('Nominal USB VBUS', 'mV', 'PA6'),
    ('NTC resistance', 'ohm', 'PC0'),
    ('ADC sample groups', 'count', ''),
    ('ADC errors', 'count', ''),
    ('Temperature', 'C', 'PC0'),
]


def display_sensor(page, index, value):
    if page == 0 and index == 8:
        return 'Uncalibrated / unavailable' if value == -2147483648 else f'{value / 1000:.2f}'
    if page == 0 and index == 5 and value == 4294967295:
        return 'Open / unavailable'
    return str(value)


class DebugGUI:
    def __init__(self, root, args):
        self.root = root
        self.thread = None
        self.stop = threading.Event()
        self.events = queue.Queue(maxsize=2)
        self.closing = False
        self.last_sample = None
        self.schema = json.loads((ROOT / 'docs/config_schema.json').read_text())
        self.transport = tk.StringVar(value='USB CDC' if args.usb else 'CANable2')
        self.port = tk.StringVar(value=args.port or ('COM3' if args.usb else 'COM7'))
        self.node = tk.StringVar(value=str(args.node))
        self.interval = tk.StringVar(value='200')
        self.connection = tk.StringVar(value='Disconnected')
        self.status = tk.StringVar(value='Read-only: no ARM, motor commands, configuration writes or saves.')
        self.config_status = tk.StringVar(value='Active RAM configuration: not read')
        root.title('Krabby — live sensors and configuration')
        root.geometry('1180x790')
        root.minsize(980, 680)
        bar = ttk.Frame(root, padding=8)
        bar.pack(fill='x')
        for label, variable, values, width in (
            ('Interface', self.transport, ('USB CDC', 'CANable2'), 12),
            ('Port', self.port, tuple(p.device for p in list_ports.comports()), 14),
            ('Node', self.node, tuple(str(i) for i in range(8)), 4),
        ):
            ttk.Label(bar, text=label + (' (CAN)' if label == 'Node' else '')).pack(side='left', padx=(0, 4))
            widget = ttk.Combobox(bar, textvariable=variable, values=values, width=width,
                                 state='normal' if label == 'Port' else 'readonly')
            widget.pack(side='left', padx=(0, 12))
            if label == 'Interface':
                widget.bind('<<ComboboxSelected>>', self.change_transport)
        ttk.Label(bar, text='Poll ms').pack(side='left')
        ttk.Spinbox(bar, from_=100, to=5000, increment=100, textvariable=self.interval, width=6).pack(side='left', padx=6)
        self.button = ttk.Button(bar, text='Connect', command=self.toggle)
        self.button.pack(side='left', padx=8)
        ttk.Label(bar, textvariable=self.connection).pack(side='left', padx=8)
        self.mcu_uid=tk.StringVar(value='Not read')
        identity=ttk.Frame(root,padding=(8,0,8,6))
        identity.pack(fill='x')
        ttk.Label(identity,text='MCU unique ID (last connected): ').pack(side='left')
        ttk.Label(identity,textvariable=self.mcu_uid).pack(side='left')
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=8)
        live = ttk.Frame(notebook, padding=8)
        notebook.add(live, text='Live sensors')
        self.sensor_tables = {}
        for column in range(3):
            live.columnconfigure(column, weight=1)
            frame = ttk.LabelFrame(live, text=f'Motor {column + 1}', padding=5)
            frame.grid(row=0, column=column, sticky='nsew', padx=3, pady=3)
            rows = [(name, '--', unit, pins[column]) for name, unit, pins in MOTOR_ROWS]
            self.sensor_tables[column + 1] = self.table(frame, ('Measurement', 'Value', 'Units', 'Pin'), rows)
        board = ttk.LabelFrame(live, text='Board sensors', padding=5)
        board.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=8)
        self.sensor_tables[0] = self.table(board, ('Measurement', 'Value', 'Units', 'Pin'),
                                          [(name, '--', unit, pin) for name, unit, pin in BOARD_ROWS])
        ttk.Label(live, textvariable=self.status, wraplength=1100).grid(row=2, column=0, columnspan=3, sticky='w', pady=5)
        config = ttk.Frame(notebook, padding=8)
        notebook.add(config, text='Configuration pages')
        ttk.Label(config, textvariable=self.config_status).pack(anchor='w', pady=(0, 8))
        ttk.Label(config, text='Active RAM values; refreshed every 2 seconds. Saved flash may differ.').pack(anchor='w')
        pages = ttk.Notebook(config)
        pages.pack(fill='both', expand=True, pady=8)
        self.config_tables = []
        for page in range(4):
            frame = ttk.Frame(pages, padding=6)
            pages.add(frame, text='Node' if page == 0 else f'Motor {page}')
            fields = {field['id']: field for field in self.schema['node' if page == 0 else 'motors']}
            rows = []
            for field_id in range(12 if page==0 else 14):
                field = fields.get(field_id, dict(name='reserved', units='', default=0, minimum=0, maximum=0))
                rows.append((field_id, field_id * 4, field['name'], '--', field['units'],
                             field['default'], f"{field['minimum']} .. {field['maximum']}"))
            self.config_tables.append(self.table(frame, ('Field', 'Offset', 'Parameter', 'Value', 'Units', 'Factory', 'Range'), rows, height=14))
        ttk.Label(root, text='Current/voltage conversions are nominal. Encoder sign, motor calibration and loaded behavior require bench validation.', padding=8).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.pump)

    @staticmethod
    def table(parent, columns, rows, height=9):
        table = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        for column in columns:
            table.heading(column, text=column)
            width = 180 if column in ('Measurement', 'Parameter') else 110 if column in ('Units', 'Range') else 75
            table.column(column, width=width, minwidth=40, stretch=True)
        for index, row in enumerate(rows):
            table.insert('', 'end', iid=str(index), values=row)
        table.pack(fill='both', expand=True)
        return table

    def change_transport(self, _event=None):
        self.port.set('COM3' if self.transport.get() == 'USB CDC' else 'COM7')

    def publish(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def worker(self, port, node_id, usb, interval):
        node = None
        try:
            node = Node(port, node_id, usb)
            version = struct.unpack('<4I', node.request(VERSION))
            if version[:2] != (1, self.schema['version']):
                raise RuntimeError(f'Unsupported protocol/config version {version[:2]}')
            uid = node.uid() if version[3] & 32 else "Unavailable (older firmware)"
            config, config_time = None, 0
            while not self.stop.is_set():
                started = time.monotonic()
                status = node.status()
                sensors = []
                for page in range(4):
                    if self.stop.is_set():
                        return
                    sensors.append(node.sensors(page))
                if started - config_time >= 2 or config is None:
                    config = []
                    for page in range(4):
                        if self.stop.is_set():
                            return
                        config.append(node.page(page))
                    config_time = time.monotonic()
                self.publish(('sample', status, sensors, config, time.monotonic(), config_time, uid))
                self.stop.wait(max(0, interval - (time.monotonic() - started)))
        except Exception as error:
            self.publish(('error', str(error)))
        finally:
            if node:
                try:
                    node.close()
                except Exception as error:
                    self.publish(('error', str(error)))

    def toggle(self):
        if self.thread and self.thread.is_alive():
            self.stop.set()
            self.connection.set('Disconnecting…')
            self.button.configure(state='disabled')
            return
        try:
            interval = int(self.interval.get())
            if not 100 <= interval <= 5000:
                raise ValueError()
            node_id = int(self.node.get())
            if not self.port.get().strip():
                raise ValueError()
        except ValueError:
            self.connection.set('Choose a port and poll interval of 100–5000 ms')
            return
        self.stop.clear()
        self.last_sample = None
        self.connection.set('Connecting…')
        self.button.configure(text='Disconnect')
        self.thread = threading.Thread(target=self.worker, args=(self.port.get().strip(), node_id,
                                       self.transport.get() == 'USB CDC', interval / 1000), daemon=True)
        self.thread.start()

    def render(self, status, sensors, config, sampled, config_time, uid=None):
        if uid is not None:self.mcu_uid.set(uid)
        self.last_sample = sampled
        for page, values in enumerate(sensors):
            table = self.sensor_tables[page]
            for index, value in enumerate(values):
                row = list(table.item(str(index), 'values'))
                row[1] = display_sensor(page, index, value)
                table.item(str(index), values=row)
        for page, values in enumerate(config):
            for index, value in enumerate(values):
                table = self.config_tables[page]
                row = list(table.item(str(index), 'values'))
                row[3] = value
                table.item(str(index), values=row)
        self.connection.set(f'Live · sample age {time.monotonic() - sampled:.1f}s')
        self.status.set(f"Node {status['node']} · uptime {status['uptime']} ms · node flags 0x{status['node_status']:02X} · "
                        f"motor flags {' / '.join(f'0x{x:02X}' for x in status['motor_status'])} · "
                        f"commands {status['commands']} · ADC errors {status['adc_errors']} · drops {status['drops']}")
        self.config_status.set(f'Active RAM configuration · age {time.monotonic() - config_time:.1f}s · schema {self.schema["version"]}')

    def pump(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'sample' and not self.stop.is_set():
                    self.render(*event[1:])
                elif event[0] == 'error':
                    self.stop.set()
                    self.connection.set('Disconnected — ' + event[1])
        except queue.Empty:
            pass
        if self.thread and self.thread.is_alive() and not self.stop.is_set() and self.last_sample is not None:
            age = time.monotonic() - self.last_sample
            self.connection.set(f"{'Live' if age < 1 else 'Waiting'} ? sample age {age:.1f}s")
        if self.thread and not self.thread.is_alive():
            self.thread = None
            self.button.configure(text='Connect', state='normal')
            if self.connection.get().startswith(('Live', 'Waiting', 'Disconnecting')):
                self.connection.set('Disconnected — displayed values are stale')
        if self.closing and self.thread is None:
            self.root.destroy()
            return
        self.root.after(50, self.pump)

    def close(self):
        self.closing = True
        self.stop.set()
        self.button.configure(state='disabled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port')
    parser.add_argument('--node', type=int, choices=range(8), default=0)
    parser.add_argument('--usb', action='store_true', help='Use USB CDC (default COM3) instead of CANable2 (COM7)')
    parser.add_argument('--smoke-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = tk.Tk()
    if args.smoke_test:
        root.withdraw()
    app = DebugGUI(root, args)
    if args.smoke_test:
        status = dict(node=0, uptime=1, node_status=1, motor_status=[65]*3, commands=[0]*3, adc_errors=0, drops=0)
        app.render(status, [[0]*8+[-2147483648]]+[[0]*7+[125,0]]*3, [[0]*12]+[[0]*14 for _ in range(3)], time.monotonic(), time.monotonic())
        assert app.sensor_tables[0].item('8', 'values')[1] == 'Uncalibrated / unavailable'
        assert app.sensor_tables[2].item('7', 'values')[1] == '125'
        assert len(app.config_tables[0].get_children()) == 12
        root.update()
        root.destroy()
        print('PASS: GUI widgets and sensor/configuration rendering (synthetic data only)')
    else:
        root.mainloop()

if __name__ == '__main__':
    main()
