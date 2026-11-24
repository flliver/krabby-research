#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "calibration_end_detector.h"
int main(int argc, char **argv)
{
    unsigned minStill = argc > 1 ? atoi(argv[1]) : CAL_END_STILL_MIN_DRIVE_MS;
    char name[128]; int retract, leaving, n;
    static const char *names[] = {"None", "Switch", "Stop", "Stuck", "Still"};
    while (scanf("%127s %d %d %d", name, &retract, &leaving, &n) == 4)
    {
        CalibrationEndDetector d;
        d.begin(0, retract != 0, leaving != 0, minStill);
        CalibrationEnd kind = CalibrationEnd::None; int at = -1, edges = 0, armk=-1;
        for (int k = 0; k < n; ++k)
        {
            int e, i, r; if (scanf("%d %d %d", &e, &i, &r)!=3) return 1;
            if (kind != CalibrationEnd::None) continue;
            kind = d.sample(e, static_cast<uint16_t>(i), r != 0);
            if (armk<0 && d.isArmed()) armk=k;
            if (kind != CalibrationEnd::None) { at = k; edges = e; }
        }
        printf("%s -> %s at k=%d edges=%d armedAt=%d iRef=%u vRef=%ld normal=%u cur=%u/%u\n", name, names[(int)kind], at, edges,
               armk, d.currentReference(), (long)d.speedReference(), d.normalSamples(), d.endCurrent(), d.endReference());
    }
    return 0;
}
