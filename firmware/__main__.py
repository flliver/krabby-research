"""
Interactive MCU menu. Run with: python -m firmware [--debug]
"""
import sys
import logging
import time

from pynput import keyboard as pynput_keyboard

from firmware.krabby_mcu import KrabbyMCUSDK, logger

# Joint order per leg pair: LKL, LHL, LHY, RHY, RHL, RKL
JOINTS_FRONT = ["FLKL", "FLHL", "FLHY", "FRHY", "FRHL", "FRKL"]
JOINTS_LEFT = ["RLKL", "RLHL", "RLHY", "MLHY", "MLHL", "MLKL"]
JOINTS_RIGHT = ["RRKL", "RRHL", "RRHY", "MRHY", "MRHL", "MRKL"]
# Extend: Q W E R T Y  |  Retract: A S D F G H
EXTEND_KEYS = ["q", "w", "e", "r", "t", "y"]
RETRACT_KEYS = ["a", "s", "d", "f", "g", "h"]
# analogWrite is 0-255 duty; 200 is ~78% -> ~18.8 V average from a 24 V rail. 255 is ~100% duty.
JOG_PWM = 255

_pressed = set()
_quit = False


def _on_press(key):
    global _quit
    if key == pynput_keyboard.Key.esc:
        _quit = True
        return
    try:
        _pressed.add(key.char)
    except AttributeError:
        _pressed.add(key)


def _on_release(key):
    try:
        _pressed.discard(key.char)
    except AttributeError:
        _pressed.discard(key)


def is_pressed(k):
    return k in _pressed


def _log_jog(jog_cmds):
    active = {k: v for k, v in jog_cmds.items() if v != 0}
    if active:
        parts = "  ".join(f"{k} {v:+d}" for k, v in active.items())
        logger.info("JOG  %s", parts)
    else:
        logger.info("JOG  (hold)")


def main():
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    listener = pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    try:
        print("\n=== Krabby MCU — Direct key control (18 joints) ===")
        print("Extend: Q W E R T Y  |  Retract: A S D F G H")
        print("Hold 1: LEFT set  |  Hold 2: RIGHT set  |  Hold 1+2: all 18  |  No 1/2: FRONT")
        print("0: Neutral (0.5)  |  9: Auto-calibrate  |  ESC: Quit")
        print()

        prev_jog = {}

        while True:
            if _quit:
                logger.info("ESC — quitting")
                break
            if is_pressed("0"):
                mcu.send_command_joints({
                    "FLHY": 0.5, "FRHY": 0.5, "FLHL": 0.5, "FLKL": 0.5, "FRHL": 0.5, "FRKL": 0.5,
                    "RLHY": 0.5, "MLHY": 0.5, "RLHL": 0.5, "RLKL": 0.5, "MLHL": 0.5, "MLKL": 0.5,
                    "RRHY": 0.5, "MRHY": 0.5, "RRHL": 0.5, "RRKL": 0.5, "MRHL": 0.5, "MRKL": 0.5,
                })
                time.sleep(0.3)  # debounce
                continue
            if is_pressed("9"):
                print("WARNING: This will move ALL limbs to find limits.")
                mcu.send_command_calibrate()
                time.sleep(0.5)
                continue

            key1 = is_pressed("1")
            key2 = is_pressed("2")
            # No 1/2 → FRONT; 1 → LEFT; 2 → RIGHT; 1+2 → all 18
            drive_front = (not key1 and not key2) or (key1 and key2)
            jog_cmds = {}
            for i in range(6):
                if is_pressed(EXTEND_KEYS[i]):
                    pwm = JOG_PWM
                elif is_pressed(RETRACT_KEYS[i]):
                    pwm = -JOG_PWM
                else:
                    pwm = 0
                jog_cmds[JOINTS_FRONT[i]] = pwm if drive_front else 0
                jog_cmds[JOINTS_LEFT[i]] = pwm if key1 else 0
                jog_cmds[JOINTS_RIGHT[i]] = pwm if key2 else 0

            if jog_cmds != prev_jog:
                _log_jog(jog_cmds)
                prev_jog = jog_cmds.copy()

            mcu.send_commands_jog(jog_cmds)

            time.sleep(0.04)  # ~25 Hz

    except KeyboardInterrupt:
        mcu.send_command_joints_hold()
    finally:
        listener.stop()
        mcu.close()


if __name__ == "__main__":
    main()
