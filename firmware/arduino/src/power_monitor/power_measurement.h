#pragma once

#include <math.h>
#include "../units/electrical_units.h"

// Readings and acquisition success from one INA228.
struct PowerMonitorMeasurement
{
    Volts voltage{NAN};
    Amps current{NAN};
    Watts power{NAN};
    Coulombs charge{NAN};
    // True when all four reads succeeded.
    bool isValid = false;
};

