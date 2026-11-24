// Native harness around the working tree's CalibrationEndDetector.
// stdin: records "name retract leaving minStillMs n" followed by n triples "e i r".
// stdout: one line per record.
#include <cstdio>
#include <cstdint>
#include "calibration_end_detector.h"
int main()
{
    char name[256]; int retract, leaving, n; unsigned minStill;
    static const char *names[] = {"None", "Switch", "Stop", "Stuck", "Still"};
    while (scanf("%255s %d %d %u %d", name, &retract, &leaving, &minStill, &n) == 5)
    {
        CalibrationEndDetector d;
        d.begin(0, retract != 0, leaving != 0, minStill);
        CalibrationEnd kind = CalibrationEnd::None; int at = -1, edges = 0, kArm = -1; unsigned iArm = 0; long vArm = 0;
        for (int k = 0; k < n; ++k)
        {
            int e, i, r;
            if (scanf("%d %d %d", &e, &i, &r) != 3) return 2;
            if (kind != CalibrationEnd::None) continue;
            kind = d.sample(e, static_cast<uint16_t>(i), r != 0);
            if (kArm < 0 && d.isArmed()) { kArm = k; iArm = d.currentReference(); vArm = d.speedReference(); }
            if (kind != CalibrationEnd::None) { at = k; edges = e; }
        }
        printf("%s %s k=%d edges=%d armed=%d kArm=%d iArm=%u vArm=%ld iRef=%u vRef=%ld cur=%u ref=%u normal=%u\n", name, names[(int)kind], at, edges,
               d.isArmed(), kArm, iArm, vArm, d.currentReference(), (long)d.speedReference(), d.endCurrent(), d.endReference(), (unsigned)d.normalSamples());
    }
    return 0;
}
