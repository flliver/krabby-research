"""Package existing OpenBLT + LoadMeFirst builds for initial USB DFU provisioning.
Does not connect to hardware. No dependencies beyond Python's standard library.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
BASE, APP, END = 0x08000000, 0x08003000, 0x0801F000


def hex_record(address, kind, payload):
    raw = bytes((len(payload),)) + address.to_bytes(2, 'big') + bytes((kind,)) + payload
    return ':' + (raw + bytes((-sum(raw) & 255,))).hex().upper()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--boot', type=Path, default=ROOT/'build/Bootloader/krabby_boot.bin')
    p.add_argument('--app', type=Path, default=ROOT/'build/LoadMeFirst/krabby_v03_bringup.srec')
    p.add_argument('--out', type=Path, default=ROOT/'bench/load_me_first')
    a = p.parse_args()
    boot = a.boot.read_bytes()
    memory = {}
    for line in a.app.read_text().splitlines():
        if not line.startswith(('S1','S2','S3')): continue
        raw=bytes.fromhex(line[2:]);width=int(line[1])+1
        if len(raw)!=raw[0]+1 or sum(raw)&255!=255:
            raise ValueError('Invalid application S-record')
        address=int.from_bytes(raw[1:1+width],'big')
        for i,byte in enumerate(raw[1+width:-1]):
            at=address+i
            if not APP<=at<END or at in memory:
                raise ValueError('Application outside its flash region or overlapping')
            memory[at]=byte
    if not memory or min(memory)!=APP: raise ValueError('Missing application vectors')
    app=bytearray(memory.get(at,255) for at in range(APP,max(memory)+1))
    if not 8 <= len(boot) <= APP-BASE or not 0x1DC <= len(app) <= END-APP:
        raise ValueError('Image exceeds its reserved region or is truncated')
    for start, data in ((BASE, boot), (APP, app)):
        sp, reset = struct.unpack_from('<2I', data)
        if not 0x20000000 < sp <= 0x20020000 or not reset & 1 or not start <= (reset & ~1) < start+len(data):
            raise ValueError('Invalid image vectors')
    checksum = -sum(struct.unpack_from('<7I', app)) & 0xFFFFFFFF
    struct.pack_into('<I', app, 0x1D8, checksum)
    assert (sum(struct.unpack_from('<7I', app)) + struct.unpack_from('<I', app, 0x1D8)[0]) & 0xFFFFFFFF == 0
    combined = boot + bytes([255])*(APP-BASE-len(boot)) + app
    lines=[]; upper=None
    for offset in range(0,len(combined),16):
        address=BASE+offset
        if address>>16 != upper:
            upper=address>>16;lines.append(hex_record(0,4,upper.to_bytes(2,'big')))
        lines.append(hex_record(address&65535,0,combined[offset:offset+16]))
    lines.append(hex_record(0,1,b''))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'LoadMeFirst.hex').write_text('\n'.join(lines)+'\n')
    records=[]
    for offset in range(0,len(app),32):
        data=app[offset:offset+32]
        raw=bytes((len(data)+5,))+(APP+offset).to_bytes(4,'big')+data
        records.append('S3'+(raw+bytes((-1-sum(raw)&255,))).hex().upper())
    (a.out/'LoadMeFirst_app.srec').write_text('\n'.join(records)+'\n')
    manifest=dict(boot_bytes=len(boot),app_bytes=len(app),combined_bytes=len(combined),
                  app_base=hex(APP),end_exclusive=hex(BASE+len(combined)),
                  signature=hex(checksum),required_DBANK=0,
                  sha256={name:hashlib.sha256((a.out/name).read_bytes()).hexdigest()
                          for name in ('LoadMeFirst.hex','LoadMeFirst_app.srec')})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
