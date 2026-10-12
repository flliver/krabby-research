#pragma once

// Placeholder reads until sim feeds pot/IS from Isaac joint state.

inline int analogRead(int /*pin*/) { return 512; }

inline int digitalRead(int /*pin*/) { return 0; }
