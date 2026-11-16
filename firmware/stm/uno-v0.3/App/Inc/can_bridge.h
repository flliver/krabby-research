#ifndef CAN_BRIDGE_H
#define CAN_BRIDGE_H
#include <stdint.h>
#include <stddef.h>
void CAN_BridgeInit(void);
uint16_t CAN_BridgeStatus(char *out, size_t size);
/* Main-loop only. Output buffers must be at least 320 bytes. */
uint16_t CAN_BridgeRead(char *out, size_t size);
uint16_t CAN_BridgeCommand(char *line, char *out, size_t size);
#endif
