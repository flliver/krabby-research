#pragma once
#include <Arduino.h>

struct Command
{
    String name; // Joint ID: LHY, RHY, LHL, LKL, RHL, RKL
    float val;   // Linear: [0.0,1.0]
};

// Helper to clear command buffer
static inline void clearCommands(Command *cmds, size_t maxCmds)
{
    for (size_t k = 0; k < maxCmds; k++)
    {
        cmds[k].name = "";
        cmds[k].val = 0.0f;
    }
}

// Helper to get next token from line
static inline String nextTok(const String &line, int &idx, int len)
{
    while (idx < len && isspace(line[idx]))
        idx++;
    if (idx >= len)
        return String("");
    int start = idx;
    while (idx < len && !isspace(line[idx]))
        idx++;
    return line.substring(start, idx);
}

// Parse "<name> <val> [<name> <val>...]" (caller already consumed leading 'T') into a caller-provided buffer.
// Returns the number of commands parsed.
inline size_t parseCommands(const String &line, Command *cmds, size_t maxCmds)
{
    if (!cmds || maxCmds == 0)
        return 0;

    clearCommands(cmds, maxCmds);

    const int len = line.length();
    if (len == 0)
        return 0;

    int i = 0;
    size_t idx = 0;

    // Parse each <name> <val> pair until buffer full or tokens exhausted.
    while (idx < maxCmds)
    {
        String name = nextTok(line, i, len);
        if (name.length() == 0)
            break;  // clean end of input — return the pairs parsed so far (NOT an error;
                    // a command rarely fills the whole buffer, e.g. a single-joint target)
        String valStr = nextTok(line, i, len);
        if (valStr.length() == 0)
        {
            // A name with no value is malformed — drop the whole batch.
            clearCommands(cmds, maxCmds);
            return 0;
        }

        cmds[idx].name = name;
        cmds[idx].val = valStr.toFloat();
        idx++;
    }

    return idx;
}
