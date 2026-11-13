/* Reject truncated XCP packets before entering the upstream command handlers.
   FD padding is allowed. No alternate update protocol is introduced. */
#include "boot.h"
blt_bool XcpPacketReceivedHook(blt_int8u *data, blt_int8u len)
{
    static unsigned long mta;
    if (!len) return BLT_TRUE;
    unsigned needed = 1;
    switch (data[0]) {
    case 0xFF: case 0xF5: needed = 2; break;
    case 0xF6: case 0xD1: needed = 8; break;
    case 0xFE: case 0xFD: case 0xFC: case 0xD2: case 0xCF: break;
    case 0xF1: needed = 3; break;
    case 0xD0:
        needed = len >= 2 ? 2U + data[1] : 2U;
        break;
    case 0xC9: needed = 64; break;
    default: {
        blt_int8u unknown[2] = {0xFE, 0x20};
        ComTransmitPacket(unknown, 2);
        return BLT_TRUE;
    }
    }
    blt_int8u error[2] = {0xFE, 0x21};
    if (len < needed) {
        ComTransmitPacket(error, 2);
        return BLT_TRUE;
    }
    /* Only the actual MCU flash is exposed for UPLOAD. Invalid MTAs must not
       lead to a BusFault when the upstream upload command dereferences one. */
    if (data[0] == 0xF6) {
        unsigned long addr = (unsigned long)data[4] | ((unsigned long)data[5]<<8)
            | ((unsigned long)data[6]<<16) | ((unsigned long)data[7]<<24);
        if (addr < 0x08000000UL || addr >= 0x08020000UL) {
            error[1] = 0x22;
            ComTransmitPacket(error, 2);
            return BLT_TRUE;
        }
        mta = addr;
    }
    if (data[0] == 0xF5) {
        if (data[1] > 63U || mta < 0x08000000UL ||
            mta >= 0x08020000UL || data[1] > 0x08020000UL-mta) {
            error[1] = 0x22;
            ComTransmitPacket(error, 2);
            return BLT_TRUE;
        }
        mta += data[1];
    }
    if (data[0] == 0xC9 || (data[0] == 0xD0 && data[1])) {
        unsigned count = data[0] == 0xC9 ? 63U : data[1];
        if (mta < 0x08003000UL || mta >= 0x0801F000UL ||
            count > 63U || count > 0x0801F000UL-mta) {
            error[1] = 0x22;
            ComTransmitPacket(error, 2);
            return BLT_TRUE;
        }
        /* Stay synchronized with the upstream MTA after successful programming.
           A hardware write failure leaves this guard conservatively ahead. */
        mta += count;
    }
    return BLT_FALSE;
}
