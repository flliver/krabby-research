# Motor board (one 2×10 header per board)

<table border="1">
<tbody>
<tr><td>1</td><td>R_PWM 1</td><td>11</td><td>L_PWM 1</td></tr>
<tr><td>2</td><td>R_PWM 2</td><td>12</td><td>L_PWM 2</td></tr>
<tr><td>3</td><td>R_PWM 3</td><td>13</td><td>L_PWM 3</td></tr>
<tr><td>4</td><td>EN 1</td><td>14</td><td>IS 1</td></tr>
<tr><td>5</td><td>EN 2</td><td>15</td><td>IS 2</td></tr>
<tr><td>6</td><td>EN 3</td><td>16</td><td>IS 3</td></tr>
<tr><td>7</td><td>POT/HALLB 1</td><td>17</td><td>HALLA 1</td></tr>
<tr><td>8</td><td>POT/HALLB 2</td><td>18</td><td>HALLA 2</td></tr>
<tr><td>9</td><td>POT/HALLB 3</td><td>19</td><td>HALLA 3</td></tr>
<tr><td>10</td><td>VCC</td><td>20</td><td>GND</td></tr>
</tbody>
</table>

# Arduino — FL board (J1)

<table border="1">
<tbody>
<tr><td>1</td><td>R_PWM1/D2</td><td>11</td><td>L_PWM1/D3</td></tr>
<tr><td>2</td><td>R_PWM2/D4</td><td>12</td><td>L_PWM2/D5</td></tr>
<tr><td>3</td><td>R_PWM3/D6</td><td>13</td><td>L_PWM3/D7</td></tr>
<tr><td>4</td><td>EN1/D22</td><td>14</td><td>IS1/A6</td></tr>
<tr><td>5</td><td>EN2/D24</td><td>15</td><td>IS2/A7</td></tr>
<tr><td>6</td><td>EN3/D26</td><td>16</td><td>IS3/A8</td></tr>
<tr><td>7</td><td>POT/HALLB1/A0</td><td>17</td><td>HALLA1/D50</td></tr>
<tr><td>8</td><td>POT/HALLB2/A1</td><td>18</td><td>HALLA2/D51</td></tr>
<tr><td>9</td><td>POT/HALLB3/A2</td><td>19</td><td>HALLA3/D52</td></tr>
<tr><td>10</td><td>VCC</td><td>20</td><td>GND</td></tr>
</tbody>
</table>

# Arduino — FR board (J2)

<table border="1">
<tbody>
<tr><td>1</td><td>R_PWM4/D8</td><td>11</td><td>L_PWM4/D9</td></tr>
<tr><td>2</td><td>R_PWM5/D10</td><td>12</td><td>L_PWM5/D11</td></tr>
<tr><td>3</td><td>R_PWM6/D12</td><td>13</td><td>L_PWM6/D13</td></tr>
<tr><td>4</td><td>EN4/D23</td><td>14</td><td>IS4/A9</td></tr>
<tr><td>5</td><td>EN5/D25</td><td>15</td><td>IS5/A10</td></tr>
<tr><td>6</td><td>EN6/D27</td><td>16</td><td>IS6/A11</td></tr>
<tr><td>7</td><td>POT/HALLB4/A3</td><td>17</td><td>HALLA4/A12</td></tr>
<tr><td>8</td><td>POT/HALLB5/A4</td><td>18</td><td>HALLA5/A13</td></tr>
<tr><td>9</td><td>POT/HALLB6/A5</td><td>19</td><td>HALLA6/A14</td></tr>
<tr><td>10</td><td>VCC</td><td>20</td><td>GND</td></tr>
</tbody>
</table>

**Routing (Mega 2560):** FL HallA1–3 use **D50, D51, D52** (Port B, PCINT0). FR HallA4–6 use **A12, A13, A14** (Port K, PCINT2). All Serial ports remain free. EN pins are interleaved: FL even (D22, D24, D26), FR odd (D23, D25, D27).

# Arduino Mega — UART (RX / TX)

<table border="1">
<tbody>
<tr><td>Port</td><td>TX (D#)</td><td>RX (D#)</td><td>Krabby use</td></tr>
<tr><td>Serial</td><td>USB</td><td>USB</td><td>Host / programming</td></tr>
<tr><td>Serial1</td><td>18</td><td>19</td><td>Available (no pin conflicts in Rev 3)</td></tr>
<tr><td>Serial2</td><td>16</td><td>17</td><td>Follower RIGHT (<code>SERIAL_RIGHT</code>)</td></tr>
<tr><td>Serial3</td><td>14</td><td>15</td><td>Follower LEFT (<code>SERIAL_LEFT</code>)</td></tr>
</tbody>
</table>

## Firmware

The tables above match **`KRABBY_PIN_REV` 3** (Krabby Uno v0.2) in `arduino/board_pins.h`. Legacy pin differences, build flags, and flashing are documented in **`SETUP.md`**.
