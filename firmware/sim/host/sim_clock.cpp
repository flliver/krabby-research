#include "sim_clock.hpp"

#include "pin_state.hpp"

namespace krabby {
namespace sim {

namespace {

SimClock g_sim_clock;

}  // namespace

void SimClock::reset() { now_ms_ = 0; }

void SimClock::set(unsigned long ms) { now_ms_ = ms; }

void SimClock::advance(unsigned long deltaMs) { now_ms_ += deltaMs; }

unsigned long SimClock::nowMs() const { return now_ms_; }

SimClock& simClock() { return g_sim_clock; }

void resetHostSim()
{
    pinState().reset();
    simClock().reset();
}

}  // namespace sim
}  // namespace krabby
