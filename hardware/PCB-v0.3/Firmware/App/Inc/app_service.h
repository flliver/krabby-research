#ifndef APP_SERVICE_H
#define APP_SERVICE_H
#include <stddef.h>
#include <stdint.h>
void App_Init(void);
void App_Poll(void);
void App_Tick(void);
void App_EmergencySleep(void);
/* Called in main context. Versioned binary service shared by CAN and USB. */
unsigned App_Request(const uint8_t *in, unsigned len, uint8_t out[64]);
uint32_t App_CommandEpoch(void);
unsigned App_USB(char *line, char *out, size_t size, uint32_t received_epoch);
int App_CANReceive(uint32_t id, const uint8_t *data, unsigned len);
#endif
