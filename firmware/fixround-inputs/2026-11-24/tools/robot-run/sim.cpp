// Minimal plant: drives the working-tree LegCalibration against a modelled leg.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "leg_calibration.h"

struct Sink : CalibrationTextSink {
    void printFlash(const char *t) override { fputs(t, stdout); }
    void printText(const char *t) override { fputs(t, stdout); }
    void printNumber(long v) override { printf("%ld", v); }
    void printChar(char c) override { putchar(c); }
    void endLine() override { putchar('\n'); }
};

struct Joint {
    double x = 0;           // extension coordinate (edges), higher = more extended
    double retEnd = 0;      // retract end (switch) x
    double extEnd = 1069;   // extend end x
    bool extIsSwitch = true;// extend end: switch (hip) or compliant stop (knee)
    double vRun = 6.0;      // edges per 50 ms at PWM 128
    double iRun = 15.0;     // running current (IS counts)
    double iSwitch = 11.0;  // switch-open current
    double stallAt = -1e9;  // retract-direction stall point (force shortfall), x
    double stallSpan = 120; // deceleration span above stallAt
    double blockAt = 1e9;   // extend-direction obstacle x (knee curl into belly etc.)
    int pwm = 0;
    double avgIS = 0;
    bool hall = true;
};

struct Plant : LegCalibrationPlant {
    Joint j[6];
    JointSensor kind[6] = {};
    bool homed[6] = {};
    int32_t home[6] = {};
    uint32_t now = 0;
    Sink sink;
    unsigned seed = 1;
    double rnd() { seed = seed * 1103515245u + 12345u; return ((seed >> 16) & 0x7fff) / 32767.0; }
    int32_t count(uint8_t n) { return j[n].hall ? -(int32_t)lround(j[n].x) : 0; }
    void drive(uint8_t n, int pwm) override { j[n].pwm = pwm; }
    void stopAll() override { for (auto &q : j) q.pwm = 0; }
    int rawPot(uint8_t) override { return 1018; }
    int potSnapshot(uint8_t) override { return 1018; }
    bool isPotOpen(uint8_t) override { return false; }
    bool hasQuadratureHall() override { return true; }
    int32_t hallCount(uint8_t n) override { return count(n); }
    int position(uint8_t n) override { return -(count(n) - home[n]); }
    JointSensor sensor(uint8_t n) override { return kind[n]; }
    void setSensor(uint8_t n, JointSensor k) override { kind[n] = k; }
    bool isHomed(uint8_t n) override { return homed[n]; }
    void setHome(uint8_t n) override { homed[n] = true; home[n] = count(n); }
    void clearHome(uint8_t n) override { homed[n] = false; }
    bool isCurrentAbsent(uint8_t) override { return false; }
    uint16_t motorCurrentTenths(uint8_t n) override { return calibrationCurrentTenths((float)j[n].avgIS); }
    bool measuredLimit(uint8_t, bool, int &) override { return false; }
    int savedStroke(uint8_t) override { return 0; }
    uint32_t nowMs() override { return now; }
    const char *jointName(uint8_t n) override { static const char *nm[] = {"FRHY","FRHL","FRKL","FLHY","FLHL","FLKL"}; return nm[n]; }
    const char *roleLabel() override { return "FRONT"; }
    CalibrationTextSink &out() override { return sink; }
    void storeLimits(uint8_t n, int a, int b) override { printf("  [storeLimits %d: %d..%d]\n", n, a, b); }
    void saveLimits() override { printf("  [saveLimits]\n"); }

    // One 5 ms physics tick.
    void tick() {
        now += 5;
        for (int n = 0; n < 6; ++n) {
            Joint &q = j[n];
            double mag = fabs((double)q.pwm);
            double target = 0;
            if (mag > 0) {
                double v = q.vRun * (mag >= 40 ? (mag - 40) / 88.0 : 0) / 10.0; // edges per 5 ms
                double load = 0; // 0 = free running, 1 = stalled by force
                if (q.pwm < 0) { // retract
                    if (q.x > q.stallAt) {
                        double f = (q.x - q.stallAt) / q.stallSpan; if (f > 1) f = 1;
                        load = 1 - f; v *= f;
                    } else { v = 0; load = 1; }
                    if (q.x <= q.retEnd) { q.x = q.retEnd; v = 0; target = q.iSwitch; }
                    else { q.x -= v; if (q.x < q.retEnd) q.x = q.retEnd; target = q.iRun * (1 + 0.8 * load) + (rnd() - 0.5) * 8 * load; }
                } else { // extend
                    double stopX = q.extEnd < q.blockAt ? q.extEnd : q.blockAt;
                    bool sw = q.extIsSwitch && q.extEnd <= q.blockAt;
                    if (sw) {
                        if (q.x >= stopX) { q.x = stopX; v = 0; target = q.iSwitch; }
                        else { q.x += v; if (q.x > stopX) q.x = stopX; target = q.iRun; }
                    } else { // compliant stop over 20 edges
                        double d = stopX - q.x;
                        double f = d / 20.0; if (f > 1) f = 1; if (f < 0) f = 0;
                        q.x += v * f; target = q.iRun * (1 + 0.6 * (1 - f));
                    }
                }
            } else target = 0;
            // 5 ms tick, avgIS EMA alpha 0.1 per ~1 ms loop -> roughly alpha 0.4 per tick
            q.avgIS += 0.4 * (target - q.avgIS);
        }
    }
};

int main(int argc, char **argv)
{
    const char *scen = argc > 1 ? argv[1] : "A";
    int pass = argc > 2 ? atoi(argv[2]) : CAL_PASS_SEEK;
    Plant p;
    // Leg A: FRHL hip (joint 1), FRKL knee (joint 2). Leg B healthy at its ends.
    for (int n = 0; n < 6; ++n) { p.j[n].x = 0; }
    Joint &hip = p.j[1], &knee = p.j[2];
    hip.retEnd = 0; hip.extEnd = 1069; hip.extIsSwitch = true; hip.iRun = 16; hip.iSwitch = 11;
    // knee: curl = extend into compliant stop at x=0, uncurl = retract to switch at -785
    knee.retEnd = -785; knee.extEnd = 0; knee.extIsSwitch = false; knee.iRun = 15; knee.iSwitch = 7; knee.x = -200;
    { int n = 4; p.j[n].retEnd = 0; p.j[n].extEnd = 1069; p.j[n].x = 300; }
    { int n = 5; p.j[n].retEnd = -785; p.j[n].extEnd = 0; p.j[n].extIsSwitch = false; p.j[n].x = -300; }
    if (!strcmp(scen, "A")) { hip.stallAt = 878; hip.x = 880; }      // re-run after the F1 stall
    else if (!strcmp(scen, "B")) { hip.stallAt = 878; hip.x = 1000; } // start 122 edges above it
    else if (!strcmp(scen, "B2")) { hip.stallAt = 878; hip.x = 950; }
    else if (!strcmp(scen, "C")) { hip.stallAt = 878; hip.x = 1069; } // from the extend switch
    else if (!strcmp(scen, "A2")) { hip.stallAt = 878; hip.x = 905; }
    else if (!strcmp(scen, "AF")) { hip.stallAt = 878; hip.x = 880; hip.blockAt = 941; }
    else if (!strcmp(scen, "AX")) { hip.stallAt = 878; hip.x = 880; }
    else if (!strcmp(scen, "H")) { hip.x = 600; }                     // healthy
    else if (!strcmp(scen, "K")) { hip.x = 600; knee.blockAt = -150; knee.x = -400; } // knee curl blocked by belly
    p.kind[1] = p.kind[2] = p.kind[4] = p.kind[5] = JointSensor::Unknown;
    LegCalibration cal;
    cal.start(p, pass, 128);
    for (int t = 0; t < 400000 && cal.isRunning(); ++t) {
        p.tick();
        cal.update(p);
    }
    printf("== outcome %d; hip x=%.0f home(x)=%.0f homed=%d; knee x=%.0f home(x)=%.0f homed=%d\n", (int)cal.outcome(),
           hip.x, -(double)p.home[1], p.homed[1], knee.x, -(double)p.home[2], p.homed[2]);
    return 0;
}
