#include "board_pins.h"

#include "board.h"
#include "board_pins_stm32.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    board_pins_t base;
    float applied_duty[3];
} stm32_pins_t;

static stm32_pins_t s_stm32_pins;

#ifdef USE_HAL_DRIVER

#include "hw_handles.h"
#include "main.h"
#include "stm32_bringup.h"

#include <stdint.h>

#define KRABBY_STM_JOINTS 3U
#define KRABBY_STM_CAL_MAGIC 0x4B43414CU
#define KRABBY_STM_DAC_LIMIT 100U

typedef struct __attribute__((packed)) {
    uint32_t magic;
    joint_cal_t joints[KRABBY_STM_JOINTS];
    uint32_t crc;
} krabby_stm_cal_flash_t;

typedef struct {
    board_pins_t base;
    float applied_duty[KRABBY_STM_JOINTS];
    uint8_t node_id;
    uint16_t pot[KRABBY_STM_JOINTS];
    int32_t hall[KRABBY_STM_JOINTS];
    joint_cal_t cal[KRABBY_STM_JOINTS];
    bool cal_valid[KRABBY_STM_JOINTS];
    int16_t cmd_power[KRABBY_STM_JOINTS];
    bool cmd_extend[KRABBY_STM_JOINTS];
    int8_t phase[KRABBY_STM_JOINTS];
    uint8_t waiting[KRABBY_STM_JOINTS];
    uint32_t wake_at[KRABBY_STM_JOINTS];
    uint8_t pwm_started;
    uint16_t enc_previous[KRABBY_STM_JOINTS];
    uint16_t raw_adc[8];
    uint16_t filtered[8];
    int32_t filter_acc[8];
    uint32_t adc_samples;
    uint32_t last_sample_ms;
} stm32_hw_pins_t;

static stm32_hw_pins_t s_hw_pins;

static GPIO_TypeDef *const s_sleep_port[KRABBY_STM_JOINTS] = {GPIOB, GPIOB, GPIOB};
static const uint16_t s_sleep_pin[KRABBY_STM_JOINTS] = {M1_SLEEP_Pin, M2_SLEEP_Pin, M3_SLEEP_Pin};
static GPIO_TypeDef *const s_dir_port[KRABBY_STM_JOINTS] = {GPIOB, GPIOB, GPIOC};
static const uint16_t s_dir_pin[KRABBY_STM_JOINTS] = {M1_DIR_Pin, M2_DIR_Pin, M3_DIR_Pin};
static GPIO_TypeDef *const s_fault_port[KRABBY_STM_JOINTS] = {GPIOC, GPIOC, GPIOB};
static const uint16_t s_fault_pin[KRABBY_STM_JOINTS] = {M1_FAULT_Pin, M2_FAULT_Pin, M3_FAULT_Pin};
static TIM_HandleTypeDef *const s_enc_timer[KRABBY_STM_JOINTS] = {&htim2, &htim8, &htim5};
static const uint32_t s_pwm_ch[KRABBY_STM_JOINTS] = {TIM_CHANNEL_1, TIM_CHANNEL_2, TIM_CHANNEL_3};

static int stm32_joint_index(int joint) {
    return joint - 1;
}

static uint32_t krabby_crc32(const void *data, unsigned size) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = ~0U;
    while (size-- > 0U) {
        crc ^= *p++;
        for (unsigned bit = 0; bit < 8U; bit++) {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
        }
    }
    return ~crc;
}

static uint8_t krabby_read_node_switch(void) {
    uint8_t id = 0U;
    if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
        id |= 4U;
    }
    if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_4) == GPIO_PIN_RESET) {
        id |= 2U;
    }
    if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_5) == GPIO_PIN_RESET) {
        id |= 1U;
    }
    return id;
}

static int krabby_dac_setup(void) {
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        const uint32_t code = ((uint32_t)KRABBY_STM_DAC_LIMIT * 4096U * 101U + 165000U) / (3300U * 100U);
        if (code > 4095U) {
            return 0;
        }
        const uint8_t bytes[3] = {(uint8_t)(0x10U | (i << 1)), (uint8_t)(code >> 4), (uint8_t)(code << 4)};
        if (HAL_I2C_Master_Transmit(&hi2c1, 0x4CU << 1, (uint8_t *)bytes, 3, 2) != HAL_OK) {
            return 0;
        }
    }
    return 1;
}

static void krabby_adc_sample(void) {
    ADC_HandleTypeDef *adc[2] = {&hadc1, &hadc2};
    uint16_t snapshot[8];
    int good = 1;
    for (unsigned a = 0; a < 2U; a++) {
        if (HAL_ADC_Start(adc[a]) != HAL_OK) {
            good = 0;
            break;
        }
        for (unsigned rank = 0; rank < 4U; rank++) {
            uint32_t spins = 20000U;
            while (!__HAL_ADC_GET_FLAG(adc[a], ADC_FLAG_EOC) && spins > 0U) {
                spins--;
            }
            if (spins == 0U || __HAL_ADC_GET_FLAG(adc[a], ADC_FLAG_OVR)) {
                good = 0;
                break;
            }
            snapshot[a * 4U + rank] = (uint16_t)HAL_ADC_GetValue(adc[a]);
        }
        (void)HAL_ADC_Stop(adc[a]);
        if (!good) {
            break;
        }
    }
    if (!good) {
        return;
    }
    memcpy(s_hw_pins.raw_adc, snapshot, sizeof(snapshot));
    for (unsigned i = 0; i < 8U; i++) {
        const int32_t target = (int32_t)snapshot[i] * 256;
        if (s_hw_pins.adc_samples == 0U) {
            s_hw_pins.filter_acc[i] = target;
        } else {
            s_hw_pins.filter_acc[i] += (target - s_hw_pins.filter_acc[i]) / 8;
        }
        s_hw_pins.filtered[i] = (uint16_t)((s_hw_pins.filter_acc[i] + 128) / 256);
    }
    s_hw_pins.adc_samples++;
    for (unsigned j = 0; j < KRABBY_STM_JOINTS; j++) {
        s_hw_pins.pot[j] = s_hw_pins.filtered[j];
    }
}

static int krabby_read_fault(unsigned motor) {
    return HAL_GPIO_ReadPin(s_fault_port[motor], s_fault_pin[motor]) == GPIO_PIN_RESET;
}

static int krabby_due(uint32_t now, uint32_t when) {
    return (int32_t)(now - when) >= 0;
}

static int krabby_pwm_enable(void) {
    if (s_hw_pins.pwm_started) {
        return 1;
    }
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        if (HAL_TIM_PWM_Start(&htim1, s_pwm_ch[i]) != HAL_OK) {
            for (unsigned j = 0; j < KRABBY_STM_JOINTS; j++) {
                (void)HAL_TIM_PWM_Stop(&htim1, s_pwm_ch[j]);
            }
            return 0;
        }
    }
    s_hw_pins.pwm_started = 1U;
    return 1;
}

static void krabby_motor_poll(uint32_t now) {
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        int power = (int)s_hw_pins.cmd_power[i];
        const int sign = s_hw_pins.cmd_extend[i] ? 1 : -1;
        int value = sign * power;
        if (power == 0) {
            __HAL_TIM_SET_COMPARE(&htim1, s_pwm_ch[i], 0U);
            s_hw_pins.applied_duty[i] = 0.0f;
            HAL_GPIO_WritePin(s_sleep_port[i], s_sleep_pin[i], GPIO_PIN_RESET);
            s_hw_pins.waiting[i] = 0U;
            continue;
        }
        if (krabby_read_fault(i)) {
            __HAL_TIM_SET_COMPARE(&htim1, s_pwm_ch[i], 0U);
            s_hw_pins.applied_duty[i] = 0.0f;
            continue;
        }
        if (s_hw_pins.waiting[i] == 1U) {
            if (!krabby_due(now, s_hw_pins.wake_at[i])) {
                continue;
            }
            HAL_GPIO_WritePin(s_dir_port[i], s_dir_pin[i], sign > 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
            s_hw_pins.phase[i] = (int8_t)sign;
            HAL_GPIO_WritePin(s_sleep_port[i], s_sleep_pin[i], GPIO_PIN_SET);
            s_hw_pins.wake_at[i] = now + 2U;
            s_hw_pins.waiting[i] = 2U;
            continue;
        }
        if (s_hw_pins.waiting[i] == 2U) {
            if (!krabby_due(now, s_hw_pins.wake_at[i])) {
                continue;
            }
            s_hw_pins.waiting[i] = 0U;
        }
        if (s_hw_pins.phase[i] != sign || HAL_GPIO_ReadPin(s_sleep_port[i], s_sleep_pin[i]) == GPIO_PIN_RESET) {
            __HAL_TIM_SET_COMPARE(&htim1, s_pwm_ch[i], 0U);
            HAL_GPIO_WritePin(s_sleep_port[i], s_sleep_pin[i], GPIO_PIN_RESET);
            s_hw_pins.wake_at[i] = now + 2U;
            s_hw_pins.waiting[i] = 1U;
            continue;
        }
        if (!krabby_pwm_enable()) {
            continue;
        }
        const uint32_t compare = (uint32_t)(power) * (htim1.Init.Period + 1U) / 1000U;
        __HAL_TIM_SET_COMPARE(&htim1, s_pwm_ch[i], compare);
        s_hw_pins.applied_duty[i] = (float)value / 1000.0f;
    }
}

static void krabby_encoder_poll(void) {
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        const uint16_t current = (uint16_t)__HAL_TIM_GET_COUNTER(s_enc_timer[i]);
        const int32_t delta = (int16_t)(uint16_t)(current - s_hw_pins.enc_previous[i]);
        s_hw_pins.enc_previous[i] = current;
        const int64_t next = (int64_t)s_hw_pins.hall[i] + (i == 1U ? -delta : delta);
        s_hw_pins.hall[i] =
            (next < INT32_MIN) ? INT32_MIN : (next > INT32_MAX) ? INT32_MAX : (int32_t)next;
    }
}

static bool krabby_load_cal_flash(void) {
    const krabby_stm_cal_flash_t *flash = (const krabby_stm_cal_flash_t *)KRABBY_CONFIG_BASE;
    if (flash->magic != KRABBY_STM_CAL_MAGIC) {
        return false;
    }
    const uint32_t crc = krabby_crc32(flash->joints, sizeof(flash->joints));
    if (flash->crc != crc) {
        return false;
    }
    memcpy(s_hw_pins.cal, flash->joints, sizeof(s_hw_pins.cal));
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        s_hw_pins.cal_valid[i] = true;
    }
    return true;
}

static bool krabby_save_cal_flash(void) {
    krabby_stm_cal_flash_t image = {0};
    image.magic = KRABBY_STM_CAL_MAGIC;
    memcpy(image.joints, s_hw_pins.cal, sizeof(image.joints));
    image.crc = krabby_crc32(image.joints, sizeof(image.joints));

    FLASH_EraseInitTypeDef erase = {0};
    uint32_t page_error = 0U;
    erase.TypeErase = FLASH_TYPEERASE_PAGES;
    erase.Banks = FLASH_BANK_1;
    erase.Page = 31U;
    erase.NbPages = 1U;

    if (HAL_FLASH_Unlock() != HAL_OK) {
        return false;
    }
    const uint32_t primask = __get_PRIMASK();
    __disable_irq();
    HAL_StatusTypeDef status = HAL_FLASHEx_Erase(&erase, &page_error);
    for (unsigned offset = 0; offset < sizeof(image) && status == HAL_OK; offset += 8U) {
        uint64_t word = 0U;
        memcpy(&word, (const uint8_t *)&image + offset, 8U);
        status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD, KRABBY_CONFIG_BASE + offset, word);
    }
    HAL_FLASH_Lock();
    __set_PRIMASK(primask);
    return status == HAL_OK;
}

static void stm32_init(board_pins_t *self) {
    (void)self;
    memset(&s_hw_pins, 0, sizeof(s_hw_pins));
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        HAL_GPIO_WritePin(s_sleep_port[i], s_sleep_pin[i], GPIO_PIN_RESET);
    }
  if (!krabby_dac_setup()) {
        Error_Handler();
    }
    s_hw_pins.node_id = krabby_read_node_switch();
    if (s_hw_pins.node_id == 0U) {
        s_hw_pins.node_id = 1U;
    }
    if (HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED) != HAL_OK ||
        HAL_ADCEx_Calibration_Start(&hadc2, ADC_SINGLE_ENDED) != HAL_OK) {
        Error_Handler();
    }
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        if (HAL_TIM_Encoder_Start(s_enc_timer[i], TIM_CHANNEL_ALL) != HAL_OK) {
            Error_Handler();
        }
        s_hw_pins.enc_previous[i] = (uint16_t)__HAL_TIM_GET_COUNTER(s_enc_timer[i]);
    }
    krabby_adc_sample();
    (void)krabby_load_cal_flash();
    for (unsigned i = 0; i < KRABBY_STM_JOINTS; i++) {
        HAL_GPIO_WritePin(s_sleep_port[i], s_sleep_pin[i], GPIO_PIN_SET);
        s_hw_pins.wake_at[i] = HAL_GetTick() + 2U;
        s_hw_pins.waiting[i] = 2U;
    }
}

static uint8_t stm32_node_id(board_pins_t *self) {
    (void)self;
    return s_hw_pins.node_id;
}

static uint32_t stm32_now_ms(board_pins_t *self) {
    (void)self;
    return HAL_GetTick();
}

static void stm32_send_pwm(board_pins_t *self, int joint, bool direction, int power) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0 || idx >= (int)KRABBY_STM_JOINTS) {
        return;
    }
    if (power < 0) {
        power = 0;
    }
    if (power > 1000) {
        power = 1000;
    }
    s_hw_pins.cmd_extend[idx] = direction;
    s_hw_pins.cmd_power[idx] = (int16_t)power;
}

static uint16_t stm32_read_pot(board_pins_t *self, int joint) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0) {
        return 0;
    }
    return s_hw_pins.pot[idx];
}

static int32_t stm32_read_hall(board_pins_t *self, int joint) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0) {
        return 0;
    }
    return s_hw_pins.hall[idx];
}

static bool stm32_read_fault(board_pins_t *self, int joint) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0) {
        return false;
    }
    return krabby_read_fault((unsigned)idx) != 0;
}

static void stm32_clear_fault(board_pins_t *self, int joint) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0) {
        return;
    }
    HAL_GPIO_WritePin(s_sleep_port[idx], s_sleep_pin[idx], GPIO_PIN_RESET);
    HAL_GPIO_WritePin(s_sleep_port[idx], s_sleep_pin[idx], GPIO_PIN_SET);
    s_hw_pins.wake_at[idx] = HAL_GetTick() + 2U;
    s_hw_pins.waiting[idx] = 2U;
}

static bool stm32_save_cal(board_pins_t *self, int joint, const joint_cal_t *cal) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0 || cal == NULL) {
        return false;
    }
    s_hw_pins.cal[idx] = *cal;
    s_hw_pins.cal_valid[idx] = true;
    return krabby_save_cal_flash();
}

static bool stm32_load_cal(board_pins_t *self, int joint, joint_cal_t *cal) {
    (void)self;
    const int idx = stm32_joint_index(joint);
    if (idx < 0 || cal == NULL) {
        return false;
    }
    if (!s_hw_pins.cal_valid[idx]) {
        return false;
    }
    *cal = s_hw_pins.cal[idx];
    return true;
}

void board_pins_stm32_poll(leg_t *leg) {
    (void)leg;
    const uint32_t now = HAL_GetTick();
    if (now != s_hw_pins.last_sample_ms) {
        s_hw_pins.last_sample_ms = now;
        krabby_adc_sample();
        krabby_encoder_poll();
    }
    krabby_motor_poll(now);
}

board_pins_t *board_pins_stm32(void) {
    stm32_hw_pins_t *pins = &s_hw_pins;
    pins->base.init = stm32_init;
    pins->base.node_id = stm32_node_id;
    pins->base.now_ms = stm32_now_ms;
    pins->base.send_pwm = stm32_send_pwm;
    pins->base.read_pot = stm32_read_pot;
    pins->base.read_hall = stm32_read_hall;
    pins->base.read_fault = stm32_read_fault;
    pins->base.clear_fault = stm32_clear_fault;
    pins->base.save_calibration = stm32_save_cal;
    pins->base.load_calibration = stm32_load_cal;
    return &pins->base;
}

float read_applied_duty(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor > 2) {
        return 0.0f;
    }
    const stm32_hw_pins_t *pins = (const stm32_hw_pins_t *)leg->pins;
    return pins->applied_duty[motor];
}

#else /* host smoke */

static void stm32_init(board_pins_t *self) {
    (void)self;
}

static uint8_t stm32_node_id(board_pins_t *self) {
    (void)self;
    return 1u;
}

static uint32_t stm32_now_ms(board_pins_t *self) {
    (void)self;
    return 0u;
}

static void stm32_send_pwm(board_pins_t *self, int joint, bool direction, int power) {
    stm32_pins_t *pins = (stm32_pins_t *)self;
    if (joint < 1 || joint > 3) {
        return;
    }
    float signed_duty = (float)power / 1000.0f;
    if (!direction) {
        signed_duty = -signed_duty;
    }
    pins->applied_duty[joint - 1] = signed_duty;
}

static uint16_t stm32_read_pot(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return 0;
}

static int32_t stm32_read_hall(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return 0;
}

static bool stm32_read_fault(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return false;
}

static void stm32_clear_fault(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
}

static bool stm32_save_cal(board_pins_t *self, int joint, const joint_cal_t *cal) {
    (void)self;
    (void)joint;
    (void)cal;
    return false;
}

static bool stm32_load_cal(board_pins_t *self, int joint, joint_cal_t *cal) {
    (void)self;
    (void)joint;
    (void)cal;
    return false;
}

void board_pins_stm32_poll(leg_t *leg) {
    (void)leg;
}

board_pins_t *board_pins_stm32(void) {
    stm32_pins_t *pins = &s_stm32_pins;
    pins->base.init = stm32_init;
    pins->base.node_id = stm32_node_id;
    pins->base.now_ms = stm32_now_ms;
    pins->base.send_pwm = stm32_send_pwm;
    pins->base.read_pot = stm32_read_pot;
    pins->base.read_hall = stm32_read_hall;
    pins->base.read_fault = stm32_read_fault;
    pins->base.clear_fault = stm32_clear_fault;
    pins->base.save_calibration = stm32_save_cal;
    pins->base.load_calibration = stm32_load_cal;
    return &pins->base;
}

float read_applied_duty(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor > 2) {
        return 0.0f;
    }
    const stm32_pins_t *pins = (const stm32_pins_t *)leg->pins;
    return pins->applied_duty[motor];
}

#endif