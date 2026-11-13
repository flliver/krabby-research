"""Read-only Krabby application discovery through CANable at 1 Mbps / 5 Mbps FD.
Only VERSION requests are sent. Does not discover boards waiting in OpenBLT,
and cannot distinguish multiple boards using the same node ID.
Close other programs using the adapter's serial port before running.
"""
import argparse
import struct
import sys

import serial
from krabby import Node, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM7', help='CANable serial port (default COM7)')
    args = parser.parse_args()
    connection = Node(args.port)
    found = []
    try:
        print(f'Scanning application nodes 0-7 on {args.port} (1M/5M CAN FD)...')
        for number in range(8):
            connection.node = number
            try:
                payload = connection.request(VERSION)
            except TimeoutError:
                print(f'Node {number}: no response')
                continue
            if len(payload) != 16:
                raise RuntimeError(f'Node {number}: unexpected VERSION length {len(payload)}')
            protocol, schema, app_base, capabilities = struct.unpack('<4I', payload)
            found.append(number)
            print(f'Node {number}: FOUND  protocol={protocol} schema={schema} '
                  f'app=0x{app_base:08X} capabilities=0x{capabilities:X}')
    finally:
        connection.close()
    print('Responding node IDs: ' + (', '.join(map(str, found)) if found else 'none'))


if __name__ == '__main__':
    try:
        main()
    except (serial.SerialException, OSError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
