/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @version        : v3.0_Cube
  * @brief          : Usb device for Virtual Com Port.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc_if.h"

/* USER CODE BEGIN INCLUDE */
#include "main.h"
#include "can_bridge.h"
#include "app_boot.h"
#include "app_service.h"
#include "board.h"

/* USER CODE END INCLUDE */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

#define RX_RING_SIZE 512U
static uint8_t rx_ring[RX_RING_SIZE];
static uint32_t rx_epoch[RX_RING_SIZE];
static volatile uint16_t rx_head, rx_tail;
static volatile uint8_t rx_overflow;
static volatile uint8_t tx_active;
static volatile uint8_t dtr, greeting_pending;
static volatile uint32_t session;
static uint8_t line_coding[7] = {0x00, 0xc2, 0x01, 0x00, 0, 0, 8};
static uint8_t led_controlled;

/* USER CODE END PV */

/** @addtogroup STM32_USB_OTG_DEVICE_LIBRARY
  * @brief Usb device library.
  * @{
  */

/** @addtogroup USBD_CDC_IF
  * @{
  */

/** @defgroup USBD_CDC_IF_Private_TypesDefinitions USBD_CDC_IF_Private_TypesDefinitions
  * @brief Private types.
  * @{
  */

/* USER CODE BEGIN PRIVATE_TYPES */

/* USER CODE END PRIVATE_TYPES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Defines USBD_CDC_IF_Private_Defines
  * @brief Private defines.
  * @{
  */

/* USER CODE BEGIN PRIVATE_DEFINES */
/* USER CODE END PRIVATE_DEFINES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Macros USBD_CDC_IF_Private_Macros
  * @brief Private macros.
  * @{
  */

/* USER CODE BEGIN PRIVATE_MACRO */

/* USER CODE END PRIVATE_MACRO */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Variables USBD_CDC_IF_Private_Variables
  * @brief Private variables.
  * @{
  */
/* Create buffer for reception and transmission           */
/* It's up to user to redefine and/or remove those define */
/** Received data over USB are stored in this buffer      */
uint8_t UserRxBufferFS[APP_RX_DATA_SIZE];

/** Data to send over USB CDC are stored in this buffer   */
uint8_t UserTxBufferFS[APP_TX_DATA_SIZE];

/* USER CODE BEGIN PRIVATE_VARIABLES */

/* USER CODE END PRIVATE_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Exported_Variables USBD_CDC_IF_Exported_Variables
  * @brief Public variables.
  * @{
  */

extern USBD_HandleTypeDef hUsbDeviceFS;

/* USER CODE BEGIN EXPORTED_VARIABLES */

/* USER CODE END EXPORTED_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_FunctionPrototypes USBD_CDC_IF_Private_FunctionPrototypes
  * @brief Private functions declaration.
  * @{
  */

static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);
static int8_t CDC_TransmitCplt_FS(uint8_t *pbuf, uint32_t *Len, uint8_t epnum);

/* USER CODE BEGIN PRIVATE_FUNCTIONS_DECLARATION */

/* USER CODE END PRIVATE_FUNCTIONS_DECLARATION */

/**
  * @}
  */

USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS,
  CDC_TransmitCplt_FS
};

/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Initializes the CDC media low layer over the FS USB IP
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Init_FS(void)
{
  /* USER CODE BEGIN 3 */
  rx_head = rx_tail = 0;
  rx_overflow = tx_active = dtr = greeting_pending = 0;
  ++session;
  /* Set Application Buffers */
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  return (USBD_OK);
  /* USER CODE END 3 */
}

/**
  * @brief  DeInitializes the CDC media low layer
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_DeInit_FS(void)
{
  /* USER CODE BEGIN 4 */
  rx_head = rx_tail = 0;
  rx_overflow = tx_active = dtr = greeting_pending = 0;
  ++session;
  return (USBD_OK);
  /* USER CODE END 4 */
}

/**
  * @brief  Manage the CDC class requests
  * @param  cmd: Command code
  * @param  pbuf: Buffer containing command data (request parameters)
  * @param  length: Number of data to be sent (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  /* USER CODE BEGIN 5 */
  switch(cmd)
  {
    case CDC_SEND_ENCAPSULATED_COMMAND:

    break;

    case CDC_GET_ENCAPSULATED_RESPONSE:

    break;

    case CDC_SET_COMM_FEATURE:

    break;

    case CDC_GET_COMM_FEATURE:

    break;

    case CDC_CLEAR_COMM_FEATURE:

    break;

  /*******************************************************************************/
  /* Line Coding Structure                                                       */
  /*-----------------------------------------------------------------------------*/
  /* Offset | Field       | Size | Value  | Description                          */
  /* 0      | dwDTERate   |   4  | Number |Data terminal rate, in bits per second*/
  /* 4      | bCharFormat |   1  | Number | Stop bits                            */
  /*                                        0 - 1 Stop bit                       */
  /*                                        1 - 1.5 Stop bits                    */
  /*                                        2 - 2 Stop bits                      */
  /* 5      | bParityType |  1   | Number | Parity                               */
  /*                                        0 - None                             */
  /*                                        1 - Odd                              */
  /*                                        2 - Even                             */
  /*                                        3 - Mark                             */
  /*                                        4 - Space                            */
  /* 6      | bDataBits  |   1   | Number Data bits (5, 6, 7, 8 or 16).          */
  /*******************************************************************************/
    case CDC_SET_LINE_CODING:
      if (length == sizeof(line_coding))
        memcpy(line_coding, pbuf, sizeof(line_coding));
    break;

    case CDC_GET_LINE_CODING:
      memcpy(pbuf, line_coding, sizeof(line_coding));
    break;

    case CDC_SET_CONTROL_LINE_STATE:
    {
      /* For zero-length requests the middleware passes the setup request. */
      const USBD_SetupReqTypedef *req = (const USBD_SetupReqTypedef *)pbuf;
      uint8_t new_dtr = (uint8_t)(req->wValue & 1U);
      if (new_dtr && !dtr)
        greeting_pending = 1;
      dtr = new_dtr;
    }
    break;

    case CDC_SEND_BREAK:

    break;

  default:
    break;
  }

  return (USBD_OK);
  /* USER CODE END 5 */
}

/**
  * @brief  Data received over USB OUT endpoint are sent over CDC interface
  *         through this function.
  *
  *         @note
  *         This function will issue a NAK packet on any OUT packet received on
  *         USB endpoint until exiting this function. If you exit this function
  *         before transfer is complete on CDC interface (ie. using DMA controller)
  *         it will result in receiving more data while previous ones are still
  *         not sent.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  /* USER CODE BEGIN 6 */
  /* Single producer (USB IRQ), single consumer (main). Never parse here. */
  for (uint32_t i = 0; i < *Len; ++i)
  {
    uint16_t next = (uint16_t)((rx_head + 1U) % RX_RING_SIZE);
    if (next == rx_tail)
    {
      rx_overflow = 1;
      break;
    }
    rx_ring[rx_head] = Buf[i];
    rx_epoch[rx_head] = App_CommandEpoch();
    rx_head = next;
  }
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
  /* USER CODE END 6 */
}

/**
  * @brief  CDC_Transmit_FS
  *         Data to send over USB IN endpoint are sent over CDC interface
  *         through this function.
  *         @note
  *
  *
  * @param  Buf: Buffer of data to be sent
  * @param  Len: Number of data to be sent (in bytes)
  * @retval USBD_OK if all operations are OK else USBD_FAIL or USBD_BUSY
  */
uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 7 */
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  /* Caller must serialize against USB deinitialization (AppPoll does). */
  if (hUsbDeviceFS.dev_state != USBD_STATE_CONFIGURED || hcdc == NULL)
    return USBD_FAIL;
  if (hcdc->TxState != 0){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
  /* USER CODE END 7 */
  return result;
}

/**
  * @brief  CDC_TransmitCplt_FS
  *         Data transmitted callback
  *
  *         @note
  *         This function is IN transfer complete callback used to inform user that
  *         the submitted Data is successfully sent over USB.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_TransmitCplt_FS(uint8_t *Buf, uint32_t *Len, uint8_t epnum)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 13 */
  UNUSED(Buf);
  UNUSED(Len);
  UNUSED(epnum);
  tx_active = 0;
  /* USER CODE END 13 */
  return result;
}

/* USER CODE BEGIN PRIVATE_FUNCTIONS_IMPLEMENTATION */

uint8_t CDC_LED_IsControlled(void)
{
  return led_controlled;
}

void CDC_AppPoll(void)
{
  static char line[256];
  static uint16_t used;
  static uint32_t line_epoch;
  static uint8_t discard;
  /* This buffer stays unchanged until the IN completion callback. */
  static char response[320];
  static uint16_t pending;
  static uint8_t submitted;
  static uint32_t seen_session;
  uint32_t primask = __get_PRIMASK();
  int ch = -1;
  uint32_t ch_epoch = 0;

  __disable_irq();
  if (seen_session != session)
  {
    seen_session = session;
    used = pending = submitted = discard = 0;
  }
  if (submitted && !tx_active)
  {
    pending = submitted = 0;
  }
  if (pending)
  {
    if (!submitted)
    {
      tx_active = 1;
      if (CDC_Transmit_FS((uint8_t *)response, pending) == USBD_OK)
        submitted = 1;
      else
        tx_active = 0; /* Keep the response and retry next iteration. */
    }
    __set_PRIMASK(primask);
    return;
  }
  if (hUsbDeviceFS.dev_state != USBD_STATE_CONFIGURED)
  {
    __set_PRIMASK(primask);
    return;
  }
  if (rx_overflow)
  {
    rx_overflow = 0;
    rx_tail = rx_head;
    used = 0;
    discard = 1; /* Never execute a command with missing bytes. */
    __set_PRIMASK(primask);
    pending = (uint16_t)snprintf(response, sizeof(response),
                               "ERR RX overflow; resend after newline\r\n");
    return;
  }
  if (greeting_pending)
  {
    greeting_pending = 0;
    __set_PRIMASK(primask);
    pending = (uint16_t)snprintf(response, sizeof(response),
                               "Krabby v0.3 USB CDC ready\r\n");
    return;
  }
  if (rx_tail != rx_head)
  {
    ch = rx_ring[rx_tail];
    ch_epoch = rx_epoch[rx_tail];
    rx_tail = (uint16_t)((rx_tail + 1U) % RX_RING_SIZE);
  }
  __set_PRIMASK(primask);

  if (ch < 0)
  {
    pending = CAN_BridgeRead(response, sizeof(response));
    return;
  }
  if (ch == '\r' || ch == '\n')
  {
    if (discard)
    {
      discard = 0;
      used = 0;
      return;
    }
    if (!used)
      return; /* Includes the LF half of CRLF. */
    line[used] = '\0';
    if (strcmp(line, "hello") == 0)
      pending = (uint16_t)snprintf(response, sizeof(response),
                                 "Hello from Krabby v0.3 | %s | toggle=%u ms\r\n",
                                 APP_IDENTITY, (unsigned)APP_BLINK_MS);
    else if (strcmp(line, "status") == 0)
      pending = (uint16_t)snprintf(response, sizeof(response),
                                 "STM32G474RBT3 uptime=%lu ms blink_ms=%u node=%u VTOR=%08lX\r\n",
                                 (unsigned long)HAL_GetTick(), (unsigned)APP_BLINK_MS,
                                 (unsigned)Board_GetNodeId(), (unsigned long)SCB->VTOR);
    else if (strcmp(line, "led on") == 0 || strcmp(line, "led off") == 0)
    {
#ifdef LOAD_ME_FIRST
      pending = (uint16_t)snprintf(response, sizeof(response), "ERR LoadMeFirst owns LED sequence\r\n");
#else
      uint8_t on = (uint8_t)(strcmp(line, "led on") == 0);
      led_controlled = 1;
      HAL_GPIO_WritePin(STATUS_LED_GPIO_Port, STATUS_LED_Pin,
                        on ? GPIO_PIN_SET : GPIO_PIN_RESET);
      pending = (uint16_t)snprintf(response, sizeof(response), "OK LED %s\r\n", on ? "on" : "off");
#endif
    }
    else if (strcmp(line, "can status") == 0)
      pending = CAN_BridgeStatus(response, sizeof(response));
    else if (strncmp(line, "tx", 2) == 0)
      pending = CAN_BridgeCommand(line, response, sizeof(response));
    else if (strncmp(line, "app ", 4) == 0 || strncmp(line, "node ", 5) == 0)
      pending = (uint16_t)App_USB(line, response, sizeof(response), line_epoch);
    else
      pending = (uint16_t)snprintf(response, sizeof(response), "ERR unknown command\r\n");
    used = 0;
  }
  else if (!discard)
  {
    if (ch == 0 || used >= sizeof(line) - 1U)
    {
      used = 0;
      discard = 1;
      pending = (uint16_t)snprintf(response, sizeof(response), "ERR invalid or too long command\r\n");
    }
    else
    {
      if (!used) line_epoch = ch_epoch;
      line[used++] = (char)ch;
    }
  }
}

/* USER CODE END PRIVATE_FUNCTIONS_IMPLEMENTATION */

/**
  * @}
  */

/**
  * @}
  */
