// Drives the working tree's LegCalibration (SEEK, leg 0) with a plant that plays real robot traces.
// Hip model: a curl plays the tail of a recorded curl trace from the frame whose remaining edges
// match the hip's distance from that trace's end point (the stall, or the real switch); an uncurl
// plays a recorded uncurl start. Knee: one recorded curl. Frames are 51.5 ms apart; the count is
// floor-interpolated, the current linearly (plus a sub-count offset for the telemetry truncation).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include "src/actuator/leg_calibration.h"

struct Trace { std::vector<int> e, i; };
static std::map<std::string, Trace> traces;
static const double FRAME_MS = 51.5;

struct Out final : public CalibrationTextSink
{
    std::string line;
    void printFlash(const char *v) override { line += v; }
    void printText(const char *v) override { line += v; }
    void printNumber(long v) override { line += std::to_string(v); }
    void printChar(char v) override { line += v; }
    void endLine() override { printf("    %s\n", line.c_str()); line.clear(); }
};

struct Plant final : public LegCalibrationPlant
{
    uint32_t now = 0;
    Out output;
    JointSensor kind[6] = {JointSensor::Unknown, JointSensor::Hall, JointSensor::Hall,
                           JointSensor::Unknown, JointSensor::Hall, JointSensor::Hall};
    int32_t count[6] = {};     // raw count at the start of the current drive
    int32_t home[6] = {};
    bool homed[6] = {};
    int pwm[6] = {};
    uint32_t t0[6] = {};
    bool driving[6] = {};
    const Trace *hipCurl = nullptr, *hipUncurl = nullptr, *kneeCurl = nullptr;
    int hipOut = 0;            // edges out from the curl trace's end point
    int hipF0 = 0, hipBase = 0;
    int offTenths = 5;
    bool kneePlayed = false;

    static int edgesAt(const Trace &t, double f, int base)
    {
        if (f < 0) return 0;
        const size_t n = t.e.size();
        size_t j = static_cast<size_t>(f);
        if (j >= n - 1) return t.e[n - 1] - base;
        const double a = f - j;
        return static_cast<int>(std::floor(t.e[j] + a * (t.e[j + 1] - t.e[j]))) - base;
    }
    static int curAt(const Trace &t, double f, int off)
    {
        const size_t n = t.i.size();
        if (f < 0) f = 0;
        size_t j = static_cast<size_t>(f);
        if (j >= n - 1) return 10 * t.i[n - 1] + off;
        const double a = f - j;
        return static_cast<int>(std::lround(10.0 * (t.i[j] + a * (t.i[j + 1] - t.i[j])))) + off;
    }
    // The trace of joint j's current drive (f: fractional frame, base: edges at its start), or null.
    const Trace *playing(uint8_t j, double &f, int &base) const
    {
        if (!driving[j]) return nullptr;
        const double frames = (now - t0[j]) / FRAME_MS;
        if (j == 1 && pwm[j] < 0 && hipCurl) { f = hipF0 + frames - 1.0; base = hipBase; return hipCurl; }
        if (j == 1 && pwm[j] > 0 && hipUncurl) { f = frames - 1.0; base = 0; return hipUncurl; }
        if (j == 2 && pwm[j] > 0 && kneeCurl && !kneePlayed) { f = frames - 1.0; base = 0; return kneeCurl; }
        return nullptr;
    }
    int gained(uint8_t j) const
    {
        double f; int base;
        const Trace *t = playing(j, f, base);
        if (!t) return 0;
        const int g = edgesAt(*t, f, base);
        return g < 0 ? 0 : g;
    }
    void settle(uint8_t j)
    {
        if (!driving[j]) return;
        const int g = gained(j);
        if (j == 1) hipOut += pwm[j] < 0 ? -g : g;
        if (j == 2 && pwm[j] > 0 && kneeCurl) kneePlayed = true;
        count[j] += pwm[j] < 0 ? g : -g;    // -PWM retracts: the raw count rises
        driving[j] = false;
    }
    void drive(uint8_t j, int value) override
    {
        if (value != 0 && !driving[j])
        {
            driving[j] = true; t0[j] = now; pwm[j] = value;
            if (j == 1 && value < 0 && hipCurl)
            {
                const Trace &t = *hipCurl; const int total = t.e.back();
                hipF0 = 0; hipBase = 0;
                if (hipOut < total)
                    for (size_t f = 0; f < t.e.size(); ++f)
                        if (total - t.e[f] <= hipOut) { hipF0 = static_cast<int>(f) + 1; hipBase = t.e[f]; break; }
            }
        }
        pwm[j] = value;
    }
    void stopAll() override { for (uint8_t j = 0; j < 6; ++j) { settle(j); pwm[j] = 0; } }
    int rawPot(uint8_t) override { return 1016; }
    int potSnapshot(uint8_t) override { return 1016; }
    bool isPotOpen(uint8_t) override { return true; }
    bool hasQuadratureHall() override { return true; }
    int32_t hallCount(uint8_t j) override { const int g = gained(j); return count[j] + (pwm[j] < 0 ? g : -g); }
    int position(uint8_t j) override { return static_cast<int>(ACTUATOR_HALL_EXTEND_COUNT_SIGN * (hallCount(j) - home[j])); }
    JointSensor sensor(uint8_t j) override { return kind[j]; }
    void setSensor(uint8_t j, JointSensor v) override { kind[j] = v; }
    bool isHomed(uint8_t j) override { return homed[j]; }
    void setHome(uint8_t j) override { home[j] = hallCount(j); homed[j] = true; }
    void clearHome(uint8_t j) override { homed[j] = false; }
    bool isCurrentAbsent(uint8_t) override { return false; }
    uint16_t motorCurrentTenths(uint8_t j) override
    {
        double f; int base;
        const Trace *t = playing(j, f, base);
        if (!t) return pwm[j] != 0 ? 100 : 0;
        return static_cast<uint16_t>(curAt(*t, f, offTenths));
    }
    bool measuredLimit(uint8_t, bool, int &) override { return false; }
    int savedStroke(uint8_t) override { return 0; }
    uint32_t nowMs() override { return now; }
    const char *jointName(uint8_t j) override
    {
        static const char *n[6] = {"FRHY", "FRHL", "FRKL", "xxHY", "xxHL", "xxKL"};
        return n[j];
    }
    const char *roleLabel() override { return "FRONT"; }
    CalibrationTextSink &out() override { return output; }
    void storeLimits(uint8_t, int, int) override {}
    void saveLimits() override {}
};

int main(int argc, char **argv)
{
    char name[128]; int n;
    while (scanf("%127s %d", name, &n) == 2)
    {
        Trace t; t.e.resize(n); t.i.resize(n);
        for (int k = 0; k < n; ++k) if (scanf("%d %d", &t.e[k], &t.i[k]) != 2) return 2;
        traces[name] = t;
    }
    const char *hip = argc > 1 ? argv[1] : "f1";
    const int stepMs = argc > 2 ? atoi(argv[2]) : 2;
    const int quiet = argc > 3 ? atoi(argv[3]) : 0;
    for (int a = 4; a < argc; ++a)
    {
        const int start = atoi(argv[a]);
        static const int offs[3] = {0, 5, 9};
        for (int q = 0; q < (quiet ? 3 : 1); ++q)
        {
            const int off = quiet ? offs[q] : 5;
            Plant plant; LegCalibration cal;
            plant.hipCurl = &traces[hip]; plant.hipUncurl = &traces["hipout"]; plant.kneeCurl = &traces["knee"];
            plant.hipOut = start; plant.offTenths = off;
            printf("== hip curl trace %s (%d edges long); hip starts %d edges before the trace end; +0.%d count; loop %d ms\n",
                   hip, traces[hip].e.back(), start, off, stepMs);
            cal.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM);
            uint32_t guard = 0;
            while (cal.isRunning() && cal.currentStep() < 2 && guard++ < 400000) { plant.now += stepMs; cal.update(plant); }
            printf("    -> %s; hip homed=%d, resting %d edges from the trace end\n",
                   cal.isRunning() ? "leg 0 accepted" : cal.outcome() == LegCalibrationOutcome::Failed ? "FAILED" : "other",
                   plant.homed[1] ? 1 : 0, plant.hipOut);
        }
    }
    return 0;
}
