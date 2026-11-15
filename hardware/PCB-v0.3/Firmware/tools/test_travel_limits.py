"""Energized travel-limit acceptance on the user's secured three-motor fixture.
Requires explicit --hardware. Backs up RAM first, restores it on failure.
On success saves the original settings plus measured output polarity only.
"""
import argparse, json, struct, time
from pathlib import Path
from krabby import Node, SET, ARM, DISARM, SAVE, LOAD


def run(port, number):
    path=Path(__file__).resolve().parents[1]/'bench/motor_pages'
    path.mkdir(exist_ok=True)
    n=Node(port,number)
    original=[list(n.page(p)) for p in range(4)]
    (path/'physical_test_start.json').write_text(json.dumps(original,indent=2))
    records=[]; success=False; final=original
    def write(page, values):
        n.write_page(page,values)
    def pulse(motor, duty, duration=.25):
        n.request(ARM); start=time.monotonic();states=[]
        try:
            while time.monotonic()-start<duration:
                values=[0]*3;values[motor]=duty;n.motor(*values)
                time.sleep(.025);states.append(n.status())
        finally:n.request(DISARM)
        time.sleep(.06)
        return states
    def native(motor):
        s=n.sensors(motor+1)
        return s[5] if original[motor+1][1] else s[4]
    try:
        n.request(DISARM)
        node=original[0].copy();write(0,node)
        for motor in range(3):
            p=original[motor+1].copy();p[11]=0;p[12:14]=[0,0];write(motor+1,p)
        time.sleep(.12)
        mask=0
        for motor in range(3):
            before=native(motor);pulse(motor,35,.4);after=native(motor)
            delta=after-before
            assert abs(delta)>5,('No measurable movement',motor+1,before,after)
            if delta<0:mask|=1<<motor
            records.append(dict(test='electrical polarity',motor=motor+1,before=before,after=after,delta=delta))
        for motor in range(3):
            p=list(n.page(motor+1));p[12]=int(bool(mask&(1<<motor)));write(motor+1,p)
        # Check that the corrected logical command means increasing feedback.
        for motor in range(3):
            before=native(motor);states=pulse(motor,35);after=native(motor)
            assert after>before+3,('Logical positive did not increase feedback',motor+1,before,after)
            records.append(dict(test='logical positive motion',motor=motor+1,before=before,after=after))
        for motor in range(3):
            for normalized in (False,True):
                for reversed_flag in (False,True):
                    for boundary,blocked in (('lower',-30),('upper',30)):
                        # Each setup is disarmed. Never reinterpret thresholds while driving.
                        value=native(motor)
                        p=original[motor+1].copy();p[11]=1;p[12:14]=[int(reversed_flag),int(normalized)]
                        if normalized:
                            value=(value-p[7])*65535//(p[8]-p[7])
                            assert 2<value<65533,('Move fixture inside calibration before test',motor+1,value)
                            p[5:7]=[min(65534,value+500),65535] if boundary=='lower' else [0,max(1,value-500)]
                        else:
                            p[5:7]=[value+20,value+1000] if boundary=='lower' else [value-1000,value-20]
                        write(motor+1,p)
                        states=pulse(motor,blocked,.13)
                        flag=4 if boundary=='lower' else 8
                        assert all(s['motor_status'][motor]&flag and not s['motor_status'][motor]&2 for s in states),('Limit did not block',motor,normalized,reversed_flag,boundary,states)
                        records.append(dict(test='blocked outward',motor=motor+1,normalized=normalized,reversed=reversed_flag,boundary=boundary))
                        # Actual retreat uses the measured, physically correct polarity.
                        if reversed_flag==bool(mask&(1<<motor)):
                            before=native(motor);states=pulse(motor,-blocked,.2);after=native(motor)
                            assert any(s['motor_status'][motor]&2 for s in states),('Retreat inhibited',motor,boundary)
                            assert (after-before)*(-blocked)>0,('Retreat moved wrong way',motor,boundary,before,after)
                            records.append(dict(test='physical retreat',motor=motor+1,normalized=normalized,boundary=boundary,before=before,after=after))
                p=original[motor+1].copy();p[11]=0
                p[12]=int(bool(mask&(1<<motor)));p[13]=0;write(motor+1,p)
        final=[p.copy() for p in original]
        for motor in range(3):final[motor+1][12]=int(bool(mask&(1<<motor)))
        # Restore user's limits and calibration; persist only measured polarity addition.
        write(0,final[0])
        for i in range(1,4):write(i,final[i])
        write(0,final[0]);n.request(SAVE);n.request(LOAD)
        assert [list(n.page(i)) for i in range(4)]==final
        success=True
        print('PASS',len(records),'checks; measured reversal bits=',mask)
    finally:
        try:
            n.request(DISARM)
            if not success:
                write(0,original[0])
                for i in range(1,4):write(i,original[i])
                write(0,original[0])
            result=dict(success=success,records=records,final_pages=[list(n.page(i)) for i in range(4)],final_status=n.status())
            (path/'physical_results.json').write_text(json.dumps(result,indent=2))
        finally:n.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hardware',required=True);p.add_argument('--node',type=int,default=7)
    a=p.parse_args();run(a.hardware,a.node)
