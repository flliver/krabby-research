/* Shared application services. No dynamic allocation or background motor control.
 * Peripheral initialization remains CubeMX-owned; all protocol, configuration,
 * acquisition and safety work is dispatched from the cooperative main loop.
 * SysTick has a short independent watchdog path; motor register writes are atomic
 * with respect to that path. Flash writes are confined to the reserved last page.
 */
#include "app_service.h"
#include "app_boot.h"
#include "board.h"
#include "main.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern ADC_HandleTypeDef hadc1, hadc2;
extern I2C_HandleTypeDef hi2c1;
extern TIM_HandleTypeDef htim1, htim2, htim5, htim8;
extern FDCAN_HandleTypeDef hfdcan2;
#define CFG_MAGIC 0x4B524346UL
#define PAGE_SIZE 56U
#define DATA_SIZE 224U
#define CONFIG_SCHEMA 4U
#define TX_SIZE 16U
static int32_t cfg[4][14];
static volatile uint8_t armed, timed_out, hw_error, sensor_error, fresh_command;
static uint8_t no_config = 1, node, initialized, sensor_init_error;
static volatile uint8_t pwm_started;
static volatile int8_t command[3], applied[3];
static volatile uint32_t last_command, command_epoch;
static uint8_t stale_request;
static uint16_t raw[8], filtered[8], dac_code[3], previous[3];
static int32_t filter_acc[8], encoder[3];
static uint32_t sample_tick, samples, adc_errors, reboot_deadline, reboot_at;
static uint8_t reboot_transport, reboot_authorized, reboot_pending;
static int8_t phase[3];
static uint32_t wake_at[3];
static uint8_t waiting[3];
static GPIO_TypeDef *const sleep_port[3] = {GPIOB, GPIOB, GPIOB};
static const uint16_t sleep_pin[3] = {M1_SLEEP_Pin, M2_SLEEP_Pin, M3_SLEEP_Pin};
static GPIO_TypeDef *const dir_port[3] = {GPIOB, GPIOB, GPIOC};
static const uint16_t dir_pin[3] = {M1_DIR_Pin, M2_DIR_Pin, M3_DIR_Pin};
static GPIO_TypeDef *const fault_port[3] = {GPIOC, GPIOC, GPIOB};
static const uint16_t fault_pin[3] = {M1_FAULT_Pin, M2_FAULT_Pin, M3_FAULT_Pin};
static TIM_HandleTypeDef *const enc_timer[3] = {&htim2, &htim8, &htim5};
static const uint32_t pwm_ch[3] = {TIM_CHANNEL_1, TIM_CHANNEL_2, TIM_CHANNEL_3};
typedef struct {
    uint32_t magic, version, size, crc;
    int32_t data[4][14];
} Stored;
_Static_assert(sizeof(Stored) == 240, "Configuration wire layout");
typedef struct {
    uint32_t id, epoch;
    uint8_t len, data[64];
} Packet;
static Packet tx_queue[TX_SIZE], rx_queue[TX_SIZE];
static unsigned tx_head, tx_tail;
static volatile unsigned rx_head, rx_tail;
static volatile uint32_t drops;
static uint32_t telemetry_tick;
static uint8_t telemetry_phase;
static int due(uint32_t now, uint32_t when) {
    return (int32_t)(now - when) >= 0;
}
static void u32put(uint8_t *p, uint32_t v) {
    memcpy(p, &v, 4);
}
static uint32_t u32get(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static uint32_t crc32(const void *data, unsigned size) {
    const uint8_t *p = data;
    uint32_t crc = ~0UL;
    while (size--) {
        crc ^= *p++;
        for (unsigned i = 0; i < 8; i++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320UL : 0);
    }
    return ~crc;
}
/* Indicate the applied drive command, never inferred mechanical motion.
 * Negative -> left LED, positive -> right LED, neutral/inhibited -> both off. */
static void set_applied(unsigned motor, int8_t value) {
    static const Board_LED_t left[3] = {BOARD_LED_M1_LEFT, BOARD_LED_M2_LEFT,
                                       BOARD_LED_M3_LEFT};
    static const Board_LED_t right[3] = {BOARD_LED_M1_RIGHT, BOARD_LED_M2_RIGHT,
                                        BOARD_LED_M3_RIGHT};
    applied[motor] = value;
    Board_LED_Set(left[motor], value < 0);
    Board_LED_Set(right[motor], value > 0);
}

void App_EmergencySleep(void) {
    GPIOB->BRR = M1_SLEEP_Pin | M2_SLEEP_Pin | M3_SLEEP_Pin;
    /* Also clear indications on exception/reboot paths without relying on polling. */
    GPIOC->BRR = M1_LEFT_LED_CTRL_Pin | M1_RIGHT_LED_CTRL_Pin;
    GPIOA->BRR = M2_LEFT_LED_CTRL_Pin | M2_RIGHT_LED_CTRL_Pin;
    GPIOB->BRR = M3_LEFT_LED_CTRL_Pin;
    GPIOD->BRR = M3_RIGHT_LED_CTRL_Pin;
    TIM1->CCR1 = TIM1->CCR2 = TIM1->CCR3 = 0;
    CLEAR_BIT(TIM1->BDTR, TIM_BDTR_MOE);
    for (unsigned i = 0; i < 3; i++) {
        command[i] = applied[i] = 0;
        waiting[i] = 0;
    }
    armed = 0;
    fresh_command = 0;
}
static int pwm_enable(void) {
    if (pwm_started)
        return 1;
    for (unsigned i = 0; i < 3; i++) {
        if (HAL_TIM_PWM_Start(&htim1, pwm_ch[i]) != HAL_OK) {
            for (unsigned j = 0; j < 3; j++)
                HAL_TIM_PWM_Stop(&htim1, pwm_ch[j]);
            return 0;
        }
    }
    pwm_started = 1;
    return 1;
}
static void safe_outputs(int force_sleep) {
    int brake = !force_sleep && !hw_error && (!cfg[1][10] || !cfg[2][10] || !cfg[3][10]);
    for (unsigned i = 0; i < 3; i++) {
        __HAL_TIM_SET_COMPARE(&htim1, pwm_ch[i], 0);
        set_applied(i, 0);
        waiting[i] = 0;
    }
    /* Latch neutral compares before any driver may wake for braking. */
    TIM1->EGR = TIM_EGR_UG;
    if (brake && !pwm_enable()) {
        hw_error = 1;
        brake = 0;
    }
    for (unsigned i = 0; i < 3; i++)
        HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i],
                          brake && cfg[i + 1][10] == 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
    if (!brake && pwm_started) {
        for (unsigned i = 0; i < 3; i++)
            HAL_TIM_PWM_Stop(&htim1, pwm_ch[i]);
        pwm_started = 0;
    }
}
static void disarm(int force_sleep) {
    armed = 0;
    fresh_command = 0;
    command_epoch++;
    for (unsigned i = 0; i < 3; i++)
        command[i] = 0;
    safe_outputs(force_sleep);
}
void App_Tick(void) {
    if (initialized && armed && cfg[0][1] &&
        (uint32_t)(HAL_GetTick() - last_command) >= (uint32_t)cfg[0][1] * 10U) {
        timed_out = 1;
        disarm(0);
    }
}
static void defaults(unsigned page) {
    memset(cfg[page], 0, PAGE_SIZE);
    if (!page) {
        cfg[0][1] = 25;
        cfg[0][3] = 3;
        cfg[0][5] = 10000;
        cfg[0][6] = 10000;
        cfg[0][7] = 25000;
    } else {
        cfg[page][4] = 100;
        cfg[page][6] = 4095;
        cfg[page][8] = 4095;
        cfg[page][9] = 3;
    }
}
static int valid_config(const int32_t c[4][14]) {
    if (c[0][0] || c[0][1] < 0 || c[0][1] > 255 || c[0][2] < 0 || c[0][2] > 60000 ||
        (c[0][2] && c[0][2] < 10) || c[0][3] < 0 || c[0][3] > 8)
        return 0;
    if (c[0][4] < 0 || c[0][4] > 10000 || c[0][5] < 1 || c[0][5] > 1000000 || c[0][6] < 1 ||
        c[0][6] > 1000000 || c[0][7] < -50000 || c[0][7] > 150000)
        return 0;
    for (unsigned f = 8; f < 14; f++)
        if (c[0][f])
            return 0;
    for (unsigned i = 1; i < 4; i++) {
        if (c[i][0] || c[i][1] < 0 || c[i][1] > 1 || c[i][2] < 0 || c[i][2] > 1 || c[i][3] ||
            c[i][4] < 1 || c[i][4] > 3000 || c[i][5] >= c[i][6] || c[i][7] >= c[i][8] ||
            (c[i][13] && (c[i][5] < 0 || c[i][6] > 65535)) ||
            c[i][9] < 0 || c[i][9] > 8 || c[i][10] < 0 || c[i][10] > 1 || c[i][11] < 0 ||
            c[i][11] > 1 || c[i][12] < 0 || c[i][12] > 1 || c[i][13] < 0 || c[i][13] > 1)
            return 0;
    }
    return 1;
}
/* Schema 2/3 used four 12-field pages. Schema 3 kept motor flags in node masks.
 * Decode into the new layout without changing flash until explicit SAVE. */
static int decode_stored(const Stored *s, int32_t dest[4][14]) {
    if (s->magic != CFG_MAGIC)
        return 0;
    if (s->version == CONFIG_SCHEMA && s->size == DATA_SIZE) {
        if (s->crc != crc32(s->data, DATA_SIZE))
            return 0;
        memcpy(dest, s->data, DATA_SIZE);
    } else if ((s->version == 2 || s->version == 3) && s->size == 192) {
        const int32_t *old = (const int32_t *)s->data;
        if (s->crc != crc32(old, 192) || old[8] < 0 || old[8] > 7 ||
            old[9] < 0 || old[9] > 7 || (s->version == 2 && (old[8] || old[9])))
            return 0;
        memset(dest, 0, DATA_SIZE);
        for (unsigned p = 0; p < 4; p++)
            memcpy(dest[p], old + p * 12, 48);
        for (unsigned i = 0; i < 3; i++) {
            dest[i + 1][12] = (old[8] >> i) & 1;
            dest[i + 1][13] = (old[9] >> i) & 1;
        }
        dest[0][8] = dest[0][9] = 0;
    } else
        return 0;
    return valid_config(dest);
}
static int stored_valid(const Stored *s) {
    int32_t decoded[4][14];
    return decode_stored(s, decoded);
}
static int dac_setup(void) {
    for (unsigned i = 0; i < 3; i++) {
        /* DAC VDD=3.3V; VREF attenuated by 100k/(100k+1k). */
        uint32_t code = ((uint32_t)cfg[i + 1][4] * 4096U * 101U + 165000U) / (3300U * 100U);
        if (code > 4095)
            return 0;
        uint8_t bytes[3] = {(uint8_t)(0x10U | (i << 1)), (uint8_t)(code >> 4),
                            (uint8_t)(code << 4)};
        if (HAL_I2C_Master_Transmit(&hi2c1, 0x4C << 1, bytes, 3, 2) != HAL_OK)
            return 0;
        uint8_t readback[2];
        /* HAL memory read emits control byte plus repeated START, per TI read sequence. */
        if (HAL_I2C_Mem_Read(&hi2c1, 0x4C << 1, (uint16_t)(i << 1), I2C_MEMADD_SIZE_8BIT, readback,
                             2, 2) != HAL_OK ||
            (((unsigned)readback[0] << 4) | (readback[1] >> 4)) != code)
            return 0;
        dac_code[i] = (uint16_t)code;
    }
    return 1;
}
static int apply_config(void) {
    disarm(1);
    /* Configuration changes must not manufacture a zero-position ramp. */
    for (unsigned i = 0; i < 8; i++) {
        filter_acc[i] = (int32_t)raw[i] * 256;
        filtered[i] = raw[i];
    }
    hw_error = !dac_setup();
    return !hw_error;
}
static unsigned save_config(void) {
    if (armed)
        return 7;
    disarm(1);
    Stored image = {CFG_MAGIC, CONFIG_SCHEMA, DATA_SIZE, 0, {{0}}};
    memcpy(image.data, cfg, DATA_SIZE);
    image.crc = crc32(image.data, DATA_SIZE);
    uint32_t error = 0;
    FLASH_EraseInitTypeDef erase = {0};
    erase.TypeErase = FLASH_TYPEERASE_PAGES;
    erase.Banks = FLASH_BANK_1;
    erase.Page = 31;
    erase.NbPages = 1;
    if (HAL_FLASH_Unlock() != HAL_OK)
        return 4;
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    HAL_StatusTypeDef status = HAL_FLASHEx_Erase(&erase, &error);
    /* Program payload first, identifying header last. */
    for (unsigned offset = 16; offset < sizeof(image) && status == HAL_OK; offset += 8) {
        uint64_t word;
        memcpy(&word, (uint8_t *)&image + offset, 8);
        status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, KRABBY_CONFIG_BASE + offset, word);
    }
    for (unsigned offset = 0; offset < 16 && status == HAL_OK; offset += 8) {
        uint64_t word;
        memcpy(&word, (uint8_t *)&image + offset, 8);
        status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, KRABBY_CONFIG_BASE + offset, word);
    }
    HAL_FLASH_Lock();
    __HAL_FLASH_INSTRUCTION_CACHE_DISABLE();
    __HAL_FLASH_INSTRUCTION_CACHE_RESET();
    __HAL_FLASH_INSTRUCTION_CACHE_ENABLE();
    __HAL_FLASH_DATA_CACHE_DISABLE();
    __HAL_FLASH_DATA_CACHE_RESET();
    __HAL_FLASH_DATA_CACHE_ENABLE();
    __set_PRIMASK(primask);
    no_config = !stored_valid((const Stored *)KRABBY_CONFIG_BASE);
    return status == HAL_OK && !no_config ? 0 : 4;
}
static unsigned load_config(void) {
    if (armed)
        return 7;
    const Stored *s = (const Stored *)KRABBY_CONFIG_BASE;
    if (!stored_valid(s)) {
        no_config = 1;
        return 5;
    }
    disarm(1);
    decode_stored(s, cfg);
    no_config = 0;
    return apply_config() ? 0 : 4;
}
static int32_t feedback(unsigned i) {
    return cfg[i + 1][1] ? encoder[i] : filtered[5 + i];
}
/* Keep normalization unsaturated for limits, saturated only for telemetry. */
static int64_t normalized(unsigned i) {
    return ((int64_t)feedback(i) - cfg[i + 1][7]) * 65535 /
           ((int64_t)cfg[i + 1][8] - cfg[i + 1][7]);
}
static int64_t limit_feedback(unsigned i) {
    return cfg[i + 1][13] ? normalized(i) : feedback(i);
}
static int32_t position(unsigned i) {
    if (!cfg[i + 1][2])
        return feedback(i);
    int64_t scaled = normalized(i);
    return scaled < 0 ? 0 : scaled > 65535 ? 65535 : (int32_t)scaled;
}
static int fault(unsigned i) {
    return HAL_GPIO_ReadPin(fault_port[i], fault_pin[i]) == GPIO_PIN_RESET;
}
static unsigned motor_status(unsigned i) {
    unsigned flags = (!hw_error && !sensor_error) ? 1 : 0;
    if (applied[i])
        flags |= 2;
    if (cfg[i + 1][11]) {
        if (limit_feedback(i) <= cfg[i + 1][5])
            flags |= 4;
        if (limit_feedback(i) >= cfg[i + 1][6])
            flags |= 8;
    }
    if (fault(i))
        flags |= 32;
    if (!armed || hw_error || sensor_error || (command[i] && !applied[i]))
        flags |= 64;
    return flags;
}
static unsigned node_status(void) {
    return ((!hw_error && !sensor_error) ? 1 : 0) | (no_config ? 2 : 0) | (timed_out ? 4 : 0) |
           (armed ? 8 : 0) | (hw_error ? 16 : 0) | (sensor_error ? 32 : 0) | (drops ? 64 : 0);
}
static unsigned status_payload(uint8_t *p) {
    u32put(p, HAL_GetTick());
    p[4] = node;
    p[5] = node_status();
    for (unsigned i = 0; i < 3; i++) {
        p[6 + i] = motor_status(i);
        p[9 + i] = (uint8_t)command[i];
    }
    u32put(p + 12, samples);
    u32put(p + 16, adc_errors);
    u32put(p + 20, drops);
    return 24;
}
static unsigned sensor_payload(unsigned page, uint8_t *p) {
    if (page == 0) {
        u32put(p, raw[3]);
        u32put(p + 4, filtered[3]);
        u32put(p + 8, raw[4]);
        u32put(p + 12, filtered[4]);
        u32put(p + 16, (uint32_t)filtered[4] * 3300U * 150U / (4095U * 82U));
        u32put(p + 20, filtered[3] >= 4095
                           ? UINT32_MAX
                           : (uint32_t)((uint64_t)cfg[0][6] * filtered[3] / (4095U - filtered[3])));
        u32put(p + 24, samples);
        u32put(p + 28, adc_errors);
        int32_t temp = INT32_MIN;
        if (cfg[0][4] && filtered[3] > 0 && filtered[3] < 4095) {
            float resistance = (float)cfg[0][6] * filtered[3] / (4095 - filtered[3]);
            float kelvin = 1.0f / (1.0f / (273.15f + cfg[0][7] / 1000.0f) +
                                   logf(resistance / cfg[0][5]) / cfg[0][4]);
            float milli = (kelvin - 273.15f) * 1000.0f;
            if (isfinite(milli) && milli >= -100000 && milli <= 250000)
                temp = (int32_t)milli;
        }
        u32put(p + 32, (uint32_t)temp);
        return 36;
    }
    unsigned i = page - 1;
    u32put(p, raw[i]);
    u32put(p + 4, filtered[i]);
    u32put(p + 8, (uint32_t)filtered[i] * 3300 / 4095);
    u32put(p + 12, raw[5 + i]);
    u32put(p + 16, filtered[5 + i]);
    u32put(p + 20, (uint32_t)encoder[i]);
    u32put(p + 24, (uint32_t)position(i));
    u32put(p + 28, dac_code[i]);
    /* 1.43k IPROPI resistor and 212uA/A combined current mirror. */
    u32put(p + 32, (uint32_t)((uint64_t)filtered[i] * 3300U * 1000000U / (4095ULL * 1430U * 212U)));
    return 36;
}
static uint8_t request_transport;
unsigned App_Request(const uint8_t *in, unsigned len, uint8_t out[64]) {
    memset(out, 0, 64);
    out[0] = 1;
    out[1] = len > 1 ? in[1] : 0;
    out[2] = len > 2 ? in[2] : 0;
    unsigned result = 1, n = 0;
    if (len < 6 || in[0] != 1)
        goto finish;
    unsigned op = in[2], arg = in[3], off = in[4], count = in[5], expected = 6;
    if (op == 5)
        expected += count;
    else if (op == 11)
        expected += 3;
    else if (op == 13)
        expected += 4;
    if (expected > 64 || len < expected)
        goto finish;
    for (unsigned i = expected; i < len; i++)
        if (in[i])
            goto finish;
    if (op != 4 && op != 5 && (off || count))
        goto finish;
    if (op != 3 && op != 4 && op != 5 && op != 6 && arg)
        goto finish;
    result = 0;
    switch (op) {
    case 1:
        u32put(out + 5, 1);
        u32put(out + 9, CONFIG_SCHEMA);
        u32put(out + 13, KRABBY_APPLICATION_BASE);
        u32put(out + 17, 63);
        n = 16;
        break;
    case 14: /* Factory-programmed 96-bit silicon identifier, three LE words. */
        u32put(out + 5, HAL_GetUIDw0());
        u32put(out + 9, HAL_GetUIDw1());
        u32put(out + 13, HAL_GetUIDw2());
        n = 12;
        break;
    case 2:
        n = status_payload(out + 5);
        break;
    case 3:
        if (arg > 3) {
            result = 2;
            break;
        }
        n = sensor_payload(arg, out + 5);
        break;
    case 4:
    case 5: {
        if (arg > 3 || off % 4 || !count || count % 4 || off >= (arg ? PAGE_SIZE : 48U) ||
            count > (arg ? PAGE_SIZE : 48U) - off) {
            result = 2;
            break;
        }
        if (op == 4) {
            memcpy(out + 5, (uint8_t *)cfg[arg] + off, count);
            n = count;
            break;
        }
        int32_t next[4][14];
        memcpy(next, cfg, sizeof(cfg));
        memcpy((uint8_t *)next[arg] + off, in + 6, count);
        if (!valid_config(next)) {
            result = 2;
            break;
        }
        disarm(1);
        memcpy(cfg, next, sizeof(cfg));
        result = apply_config() ? 0 : 4;
        break;
    }
    case 6:
        if (arg > 3 && arg != 255) {
            result = 2;
            break;
        }
        disarm(1);
        if (arg == 255)
            for (unsigned i = 0; i < 4; i++)
                defaults(i);
        else
            defaults(arg);
        result = apply_config() ? 0 : 4;
        break;
    case 7:
        result = save_config();
        break;
    case 8:
        result = load_config();
        break;
    case 9:
        disarm(1);
        hw_error = !dac_setup();
        if (hw_error || sensor_error) {
            result = 4;
            break;
        }
        if (!pwm_enable()) {
            hw_error = 1;
            result = 4;
            break;
        }
        /* Publish the ARM boundary atomically after hardware setup. Packets
           arriving during DAC initialization still belong to the old epoch. */
        {
            uint32_t mask = __get_PRIMASK();
            __disable_irq();
            timed_out = 0;
            last_command = HAL_GetTick();
            command_epoch++;
            armed = 1;
            __set_PRIMASK(mask);
        }
        break;
    case 10:
        disarm(0);
        break;
    case 11:
        for (unsigned i = 0; i < 3; i++)
            if (in[6 + i] == 128) {
                result = 2;
                break;
            }
        if (result)
            break;
        if (!armed || stale_request) {
            result = 3;
            break;
        }
        /* SysTick cannot expire between command validation and timer refresh. */
        {
            uint32_t mask = __get_PRIMASK();
            __disable_irq();
            if (!armed)
                result = 3;
            else {
                last_command = HAL_GetTick();
                fresh_command = 1;
                for (unsigned i = 0; i < 3; i++)
                    command[i] = (int8_t)in[6 + i];
            }
            __set_PRIMASK(mask);
        }
        break;
    case 12:
        reboot_authorized = 1;
        reboot_deadline = HAL_GetTick() + 500;
        reboot_transport = request_transport;
        u32put(out + 5, 0x4B524242);
        n = 4;
        break;
    case 13: {
        int allowed = reboot_authorized && !due(HAL_GetTick(), reboot_deadline) &&
                      reboot_transport == request_transport && u32get(in + 6) == 0x4B524242;
        reboot_authorized = 0;
        if (!allowed) {
            result = 6;
            break;
        }
        disarm(1);
        reboot_at = HAL_GetTick() + 30;
        reboot_pending = 1;
        break;
    }
    default:
        result = 2;
        break;
    }
finish:
    out[3] = (uint8_t)result;
    out[4] = (uint8_t)n;
    return 5 + n;
}
static void adc_sample(void) {
    ADC_HandleTypeDef *adc[2] = {&hadc1, &hadc2};
    uint16_t snapshot[8];
    int good = 1;
    for (unsigned a = 0; a < 2; a++) {
        if (HAL_ADC_Start(adc[a]) != HAL_OK) {
            good = 0;
            break;
        }
        for (unsigned r = 0; r < 4; r++) {
            /* Short bounded conversion burst; interrupts stay enabled. */
            uint32_t spins = 20000;
            while (!__HAL_ADC_GET_FLAG(adc[a], ADC_FLAG_EOC) && --spins) {
            }
            if (!spins || __HAL_ADC_GET_FLAG(adc[a], ADC_FLAG_OVR)) {
                good = 0;
                break;
            }
            snapshot[a * 4 + r] = (uint16_t)HAL_ADC_GetValue(adc[a]);
        }
        HAL_ADC_Stop(adc[a]);
        if (!good)
            break;
    }
    if (!good) {
        adc_errors++;
        sensor_error = 1;
        return;
    }
    memcpy(raw, snapshot, sizeof(raw));
    for (unsigned i = 0; i < 8; i++) {
        unsigned motor = i < 3 ? i : i >= 5 ? i - 5 : 3;
        unsigned shift = motor < 3 ? (unsigned)cfg[motor + 1][9] : (unsigned)cfg[0][3];
        int32_t target = (int32_t)raw[i] * 256;
        if (!samples)
            filter_acc[i] = target;
        else
            filter_acc[i] += (target - filter_acc[i]) / (1 << shift);
        filtered[i] = (uint16_t)((filter_acc[i] + 128) / 256);
    }
    samples++;
    sensor_error = sensor_init_error;
}
static void motor_poll(uint32_t now) {
    if (!armed)
        return;
    if (hw_error || sensor_error) {
        disarm(1);
        return;
    }
    for (unsigned i = 0; i < 3; i++) {
        int value = command[i];
        int electrical = cfg[i + 1][12] ? -value : value;
        int sign = electrical < 0 ? -1 : 1;
        if (cfg[i + 1][11]) {
            int invalid = !cfg[i + 1][1] && (filtered[5 + i] == 0 || filtered[5 + i] >= 4095);
            if (invalid || (value < 0 && limit_feedback(i) <= cfg[i + 1][5]) ||
                (value > 0 && limit_feedback(i) >= cfg[i + 1][6]))
                value = 0;
        }
        if (!value) {
            __HAL_TIM_SET_COMPARE(&htim1, pwm_ch[i], 0);
            set_applied(i, 0);
            /* ARM alone stays Hi-Z. A fresh zero command follows safe policy,
               with the same wake/fault qualification used for nonzero duty. */
            if (fresh_command) {
                if (cfg[i + 1][10]) {
                    HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i], GPIO_PIN_RESET);
                    waiting[i] = 0;
                } else if (HAL_GPIO_ReadPin(sleep_port[i], sleep_pin[i]) == GPIO_PIN_RESET) {
                    HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i], GPIO_PIN_SET);
                    wake_at[i] = now + 2;
                    waiting[i] = 2;
                }
            }
            if (HAL_GPIO_ReadPin(sleep_port[i], sleep_pin[i]) == GPIO_PIN_SET &&
                (waiting[i] != 2 || due(now, wake_at[i])) && fault(i)) {
                hw_error = 1;
                disarm(1);
                return;
            }
            continue;
        }
        if (waiting[i] == 2 && phase[i] != sign) {
            __HAL_TIM_SET_COMPARE(&htim1, pwm_ch[i], 0);
            set_applied(i, 0);
            HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i], GPIO_PIN_RESET);
            waiting[i] = 1;
            wake_at[i] = now + 2;
        }
        if (waiting[i] == 1) {
            if (!due(now, wake_at[i]))
                continue;
            HAL_GPIO_WritePin(dir_port[i], dir_pin[i], sign > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
            phase[i] = (int8_t)sign;
            HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i], GPIO_PIN_SET);
            wake_at[i] = now + 2;
            waiting[i] = 2;
            continue;
        }
        if (waiting[i] == 2) {
            if (!due(now, wake_at[i]))
                continue;
            waiting[i] = 0;
        }
        if (phase[i] != sign || HAL_GPIO_ReadPin(sleep_port[i], sleep_pin[i]) == GPIO_PIN_RESET) {
            __HAL_TIM_SET_COMPARE(&htim1, pwm_ch[i], 0);
            set_applied(i, 0);
            HAL_GPIO_WritePin(sleep_port[i], sleep_pin[i], GPIO_PIN_RESET);
            wake_at[i] = now + 2;
            waiting[i] = 1;
            continue;
        }
        if (fault(i)) {
            hw_error = 1;
            disarm(1);
            return;
        }
        __HAL_TIM_SET_COMPARE(&htim1, pwm_ch[i],
                              (uint32_t)(value < 0 ? -value : value) * (htim1.Init.Period + 1U) /
                                  127U);
        set_applied(i, (int8_t)value);
    }
}
static int enqueue(uint32_t id, const uint8_t *data, unsigned len) {
    if (tx_head - tx_tail >= TX_SIZE) {
        drops++;
        return 0;
    }
    Packet *p = &tx_queue[tx_head % TX_SIZE];
    p->id = id;
    p->len = (uint8_t)len;
    memcpy(p->data, data, len);
    tx_head++;
    return 1;
}
int App_CANReceive(uint32_t id, const uint8_t *data, unsigned len) {
    if (id < 0x500 || id > 0x507)
        return 0;
    if (id != 0x500U + node)
        return 1;
    if (rx_head - rx_tail >= TX_SIZE) {
        drops++;
        return 1;
    }
    Packet *p = &rx_queue[rx_head % TX_SIZE];
    p->epoch = command_epoch;
    p->len = (uint8_t)len;
    memcpy(p->data, data, len);
    __DMB();
    rx_head++;
    return 1;
}
void App_Init(void) {
    node = Board_GetNodeId();
    App_EmergencySleep();
    for (unsigned i = 0; i < 4; i++)
        defaults(i);
    const Stored *s = (const Stored *)KRABBY_CONFIG_BASE;
    if (stored_valid(s)) {
        decode_stored(s, cfg);
        no_config = 0;
    }
    hw_error = !dac_setup();
    sensor_error = (HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED) != HAL_OK ||
                    HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED) != HAL_OK);
    for (unsigned i = 0; i < 3; i++) {
        if (HAL_TIM_Encoder_Start(enc_timer[i], TIM_CHANNEL_ALL) != HAL_OK)
            sensor_error = 1;
        previous[i] = (uint16_t)__HAL_TIM_GET_COUNTER(enc_timer[i]);
    }
    sensor_init_error = sensor_error;
    adc_sample();
    sample_tick = HAL_GetTick();
    initialized = 1;
}
void App_Poll(void) {
    uint32_t now = HAL_GetTick();
    if (reboot_pending && due(now, reboot_at)) {
        App_EmergencySleep();
        NVIC_SystemReset();
    }
    if (now != sample_tick) {
        sample_tick = now;
        adc_sample();
        for (unsigned i = 0; i < 3; i++) {
            uint16_t current = (uint16_t)__HAL_TIM_GET_COUNTER(enc_timer[i]);
            int32_t delta = (int16_t)(uint16_t)(current - previous[i]);
            previous[i] = current;
            int64_t next = (int64_t)encoder[i] + (i == 1 ? -delta : delta);
            encoder[i] = next < INT32_MIN   ? INT32_MIN
                         : next > INT32_MAX ? INT32_MAX
                                            : (int32_t)next;
        }
    }
    App_Tick();
    /* Prevent SysTick disarm from racing a later motor register write. */
    uint32_t mask = __get_PRIMASK();
    __disable_irq();
    motor_poll(now);
    __set_PRIMASK(mask);
    if (rx_tail != rx_head) {
        __DMB();
        Packet p = rx_queue[rx_tail % TX_SIZE];
        rx_tail++;
        uint8_t reply[64];
        request_transport = 1;
        stale_request = p.epoch != command_epoch;
        unsigned len = App_Request(p.data, p.len, reply);
        stale_request = 0;
        enqueue(0x580U + node, reply, len);
    }
    if (cfg[0][2] && (uint32_t)(now - telemetry_tick) >= (uint32_t)cfg[0][2]) {
        telemetry_tick = now;
        uint8_t reply[64] = {1, 0, 2, 0, 24};
        unsigned length;
        if (telemetry_phase == 0)
            length = status_payload(reply + 5);
        else {
            reply[1] = telemetry_phase;
            reply[2] = 3;
            length = sensor_payload(telemetry_phase - 1, reply + 5);
        }
        reply[4] = (uint8_t)length;
        telemetry_phase = (telemetry_phase + 1) % 5;
        if (tx_head == tx_tail)
            enqueue(0x600U + node, reply, length + 5);
    }
    if (tx_head != tx_tail && HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan2)) {
        Packet *p = &tx_queue[tx_tail % TX_SIZE];
        static const uint8_t sizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64};
        unsigned dlc = 0;
        while (sizes[dlc] < p->len)
            dlc++;
        uint8_t wire[64] = {0};
        memcpy(wire, p->data, p->len);
        FDCAN_TxHeaderTypeDef h = {0};
        h.Identifier = p->id;
        h.IdType = FDCAN_STANDARD_ID;
        h.TxFrameType = FDCAN_DATA_FRAME;
        h.DataLength = dlc;
        h.FDFormat = FDCAN_FD_CAN;
        h.BitRateSwitch = FDCAN_BRS_ON;
        if (HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan2, &h, wire) == HAL_OK)
            tx_tail++;
    }
}
uint32_t App_CommandEpoch(void) {
    return command_epoch;
}

unsigned App_USB(char *line, char *out, size_t size, uint32_t received_epoch) {
    uint8_t request[64] = {1, 1, 0, 0, 0, 0}, response[64];
    unsigned count = 0;
    char *s = line + 4;
    if (!strncmp(line, "node ", 5)) {
        char *tokens[8], *token = strtok(line + 5, " \t");
        unsigned total = 0;
        while (token && total < 8) {
            tokens[total++] = token;
            token = strtok(NULL, " \t");
        }
        if (token || !total)
            goto bad;
        struct {
            const char *name;
            unsigned op, args;
        } const commands[] = {{"uid", 14, 0}, {"version", 1, 0},  {"status", 2, 0},  {"sensors", 3, 1},
                              {"get", 4, 2},      {"page", 4, 1},    {"set", 5, 3},
                              {"defaults", 6, 1}, {"save", 7, 0},    {"load", 8, 0},
                              {"arm", 9, 0},      {"disarm", 10, 0}, {"motor", 11, 3}};
        unsigned c;
        for (c = 0; c < sizeof(commands) / sizeof(commands[0]); c++)
            if (!strcmp(tokens[0], commands[c].name))
                break;
        if (c == sizeof(commands) / sizeof(commands[0]) || total != commands[c].args + 1)
            goto bad;
        int32_t v[3] = {0};
        for (unsigned i = 1; i < total; i++) {
            char *end;
            errno = 0;
            long x = strtol(tokens[i], &end, 0);
            if (errno || !*tokens[i] || *end)
                goto bad;
            v[i - 1] = (int32_t)x;
        }
        request[2] = (uint8_t)commands[c].op;
        count = 6;
        if (request[2] == 3 || request[2] == 4 || request[2] == 5 || request[2] == 6) {
            if (v[0] < 0 || v[0] > 255)
                goto bad;
            request[3] = (uint8_t)v[0];
        }
        if (request[2] == 4 || request[2] == 5) {
            if (total >= 3 && (v[1] < 0 || v[1] > (request[3] ? 13 : 11)))
                goto bad;
            request[4] = (uint8_t)(v[1] * 4);
            request[5] = total == 2 ? (request[3] ? PAGE_SIZE : 48U) : 4;
        }
        if (request[2] == 5) {
            u32put(request + 6, (uint32_t)v[2]);
            count = 10;
        }
        if (request[2] == 11) {
            for (unsigned i = 0; i < 3; i++) {
                if (v[i] < -127 || v[i] > 127)
                    goto bad;
                request[6 + i] = (uint8_t)v[i];
            }
            count = 9;
        }
    } else {
        if (strncmp(line, "app ", 4))
            return 0;
        while (*s) {
            while (*s == ' ')
                s++;
            if (!*s)
                break;
            if (count == 64)
                return (unsigned)snprintf(out, size, "ERR app packet too long\r\n");
            char pair[3] = {s[0], s[1], 0}, *end;
            if (!s[1])
                return (unsigned)snprintf(out, size, "ERR app hex\r\n");
            unsigned long byte = strtoul(pair, &end, 16);
            if (*end || pair[0] == '-' || pair[0] == '+')
                return (unsigned)snprintf(out, size, "ERR app hex\r\n");
            request[count++] = (uint8_t)byte;
            s += 2;
        }
    }
    request_transport = 2;
    stale_request = received_epoch != command_epoch;
    unsigned len = App_Request(request, count, response);
    stale_request = 0;
    unsigned n = (unsigned)snprintf(out, size, "APP ");
    for (unsigned i = 0; i < len && n + 4 < size; i++)
        n += (unsigned)snprintf(out + n, size - n, "%02X", response[i]);
    return n + (unsigned)snprintf(out + n, size - n, "\r\n");
bad:
    return (unsigned)snprintf(out, size, "ERR node syntax; see docs/APPLICATION_PROTOCOL.md\r\n");
}
