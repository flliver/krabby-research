"""Offline checks for committed firmware records, binaries and package manifests.
No serial ports, ST-Link, or third-party Python packages are used.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def srec_memory(path):
    memory = {}
    for number, line in enumerate(path.read_text(encoding='ascii').splitlines(), 1):
        if not line: continue
        if len(line)<4 or line[0]!='S': raise ValueError(f'{path}:{number}: bad S-record')
        raw=bytes.fromhex(line[2:])
        if len(raw)!=raw[0]+1 or sum(raw)&255!=255:
            raise ValueError(f'{path}:{number}: count/checksum mismatch')
        if line[1] not in '123': continue
        width=int(line[1])+1;address=int.from_bytes(raw[1:1+width],'big')
        for i,value in enumerate(raw[1+width:-1]):
            at=address+i
            if not 0x08000000<=at<0x08020000 or at in memory:
                raise ValueError(f'{path}:{number}: overlapping/out-of-flash data')
            memory[at]=value
    if not memory: raise ValueError(f'{path}: no firmware data')
    return memory


def ihex_memory(path):
    memory={};upper=0;ended=False
    for number,line in enumerate(path.read_text(encoding='ascii').splitlines(),1):
        if not line: continue
        if ended or not line.startswith(':'): raise ValueError(f'{path}:{number}: bad HEX record')
        raw=bytes.fromhex(line[1:])
        if len(raw)!=raw[0]+5 or sum(raw)&255: raise ValueError(f'{path}:{number}: count/checksum mismatch')
        address=int.from_bytes(raw[1:3],'big');kind=raw[3];payload=raw[4:-1]
        if kind==4 and len(payload)==2:upper=int.from_bytes(payload,'big')<<16
        elif kind==1 and not payload:ended=True
        elif kind==0:
            for i,value in enumerate(payload):
                at=upper+address+i
                if not 0x08000000<=at<0x0801F000 or at in memory:
                    raise ValueError(f'{path}:{number}: overlapping/out-of-application flash data')
                memory[at]=value
        else:raise ValueError(f'{path}:{number}: unsupported HEX record type {kind}')
    if not ended:raise ValueError(f'{path}: missing EOF')
    return memory


def main():
    counts=dict(srecords=0,binaries=0,hex_files=0,manifests=0)
    for folder in ('bench',):
        for path in (ROOT/folder).rglob('*.srec'):
            memory=srec_memory(path);counts['srecords']+=1
            binary=path.with_suffix('.bin')
            if binary.exists():
                expected=bytes(memory.get(at,255) for at in range(min(memory),max(memory)+1))
                if binary.read_bytes()!=expected:raise ValueError(f'{binary}: does not match S-record')
                counts['binaries']+=1
        for path in (ROOT/folder).rglob('*.hex'):
            ihex_memory(path);counts['hex_files']+=1
        for path in (ROOT/folder).rglob('manifest.json'):
            for name,digest in json.loads(path.read_text())['sha256'].items():
                artifact=path.parent/name
                if hashlib.sha256(artifact.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'{artifact}: manifest hash mismatch')
            counts['manifests']+=1
    # Combined provisioning package must actually contain its signed application.
    package=ROOT/'bench/load_me_first'
    app=srec_memory(package/'LoadMeFirst_app.srec')
    combined=ihex_memory(package/'LoadMeFirst.hex')
    if any(combined.get(at)!=value for at,value in app.items()):
        raise ValueError('LoadMeFirst application differs between HEX and S-record')
    base=0x08003000
    word=lambda at:int.from_bytes(bytes(app[at+i] for i in range(4)),'little')
    if (sum(word(base+offset) for offset in range(0,28,4))+word(base+0x1D8))&0xFFFFFFFF:
        raise ValueError('LoadMeFirst OpenBLT signature is invalid')
    for path in ROOT.rglob('*.bin'):
        if path.stat().st_size>0x20000:
            raise ValueError(f'{path}: exceeds physical MCU flash size')
    print('PASS offline firmware artifacts:',json.dumps(counts))


if __name__=='__main__':main()
