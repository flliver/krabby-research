"""Mutation check for the review fixes: each mutation reverts one fix in the working tree,
builds and runs the three affected native tests in Docker, and restores the file."""
import os
import subprocess
import sys

REPO = r'C:\Users\nickm\krabby\krabby-research'
DET = os.path.join(REPO, r'firmware\arduino\src\actuator\calibration_end_detector.h')
LEG = os.path.join(REPO, r'firmware\arduino\src\actuator\leg_calibration.h')
MUTATIONS = {
    'F1 iRef guard off': (DET, '(worst < stopMin_ &&\r\n                      static_cast<uint32_t>(worst) * 100UL <=',
                          '(worst < stopMin_ && 0 == 0 &&\r\n                      static_cast<uint32_t>(worst) * 100UL >= 0 * '),
    'F2 fallback off': (DET, 'if (!leaving_ && normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES &&',
                        'if (false && normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES &&'),
    'F2 retract guard off': (DET, '(!retract_ || med4 <= iRef_)', '(true || med4 <= iRef_)'),
    'F2 normal count off': (DET, 'normalSamples_ < CAL_END_SHORT_NORMAL_SAMPLES &&\r\n',
                            'normalSamples_ < 255 &&\r\n'),
    'F2 run bound off': (DET, 'ran < CAL_END_SHORT_RUN_SAMPLES &&', 'ran < 0xFFFF &&'),
    'F4 window off': (DET, 'e >= CAL_HALL_CURL_EVIDENCE_EDGES ? kStallSamples : kStillWindowSamples',
                      'e >= 30000 ? kStallSamples : kStillWindowSamples'),
    'F6 blind margin off': (LEG, '+ CAL_STALL_WINDOW_MS\r\n                : nudgeDriveMs_;', '+ 0\r\n                : nudgeDriveMs_;'),
}
TESTS = ['test_calibration_end_detector', 'test_leg_calibration', 'test_host_calibration_timeline']


def run():
    cmd = ' ; '.join(
        f'cmake --build /work/build/native-firmware-tests --target {t} >/dev/null 2>&1 || echo BUILD_FAIL {t}; '
        f'/work/build/native-firmware-tests/{t} 2>&1 | grep -E ":FAIL|Tests .* Failures"' for t in TESTS)
    env = dict(os.environ, MSYS_NO_PATHCONV='1')
    out = subprocess.run(['docker', 'run', '--rm', '-v', 'C:/Users/nickm/krabby/krabby-research:/src:ro', '-v',
                          'krabby-fwtest-work-ced:/work', 'krabby-fwtest:latest', 'sh', cmd],
                         capture_output=True, text=True, env=env)
    return out.stdout + out.stderr


names = sys.argv[1:] or list(MUTATIONS)
for name in names:
    path, old, new = MUTATIONS[name]
    src = open(path, newline='').read()
    assert src.count(old) == 1, (name, src.count(old))
    try:
        open(path, 'w', newline='').write(src.replace(old, new))
        res = run()
    finally:
        open(path, 'w', newline='').write(src)
    fails = [l for l in res.splitlines() if ':FAIL' in l or 'BUILD_FAIL' in l]
    print(f'== {name}: {"KILLED" if fails else "SURVIVED"}')
    for l in fails[:8]:
        print('   ', l[:220])
