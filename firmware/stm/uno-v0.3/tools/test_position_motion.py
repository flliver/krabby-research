"""Explicit energized PC-controller acceptance; restores original RAM configuration."""
import argparse,json,time,struct
from pathlib import Path
from krabby import Node,SET,DISARM
from krabby_position_gui import PositionController

def run(port,node):
    n=Node(port,node);original=[list(n.page(i)) for i in range(4)];c=None;events=[];results=[]
    try:
        n.request(DISARM)
        for i in range(1,4):
            p=original[i].copy();p[11]=0;n.write_page(i,p)
        n.close();n=None
        c=PositionController(port,node,False,200,events.append);c.thread.start()
        def wait(predicate,seconds=5):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if predicate():return
                if not c.thread.is_alive():raise RuntimeError(events[-5:])
                time.sleep(.025)
            raise TimeoutError(events[-5:])
        wait(lambda:any(e[0]=='connected' for e in events));c.settings(None,[35]*3,6000,500)
        token=c.enable_position();wait(lambda:c.position_active==token)
        for direction in (-1,1):
            samples=[e for e in events if e[0]=='position_sample'];before=list(samples[-1][3]);target=[max(1500,min(64000,p+direction*2000)) for p in before]
            c.settings(target,[35]*3,6000,500)
            start=len(events)
            wait(lambda:any(e[0]=='position_sample' and all(abs(a-b)<=650 for a,b in zip(e[3],target)) for e in events[start:]),12)
            samples=[e for e in events[start:] if e[0]=='position_sample'];results.append(dict(direction=direction,before=before,target=target,after=samples[-1][3]))
        c.disable_position();wait(lambda:c.position_active is None and not c.repeating)
        print('PASS actual PC position-controller motion',results)
    finally:
        if c is not None:c.stop.set();c.thread.join(4)
        if n is not None:n.close()
        n=Node(port,node)
        try:
            n.request(DISARM)
            for i,p in enumerate(original):n.write_page(i,p)
            (Path(__file__).resolve().parents[1]/'bench/motor_pages/position_motion.json').write_text(json.dumps(dict(results=results,restored=[list(n.page(i)) for i in range(4)],status=n.status()),indent=2))
        finally:n.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hardware',required=True);p.add_argument('--node',type=int,default=7);a=p.parse_args();run(a.hardware,a.node)
