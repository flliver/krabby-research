import time
import logging
import argparse
from krabby_mcu_six_axis import KrabbyMCUSDK

# Configure clean logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CalibrationWizard")


def get_user_input(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return 'q'


def calibration_wizard():
    print("\n=== Krabby-Uno Linear Actuator Calibration Wizard ===")
    print("SAFETY NOTICE: This tool uses 'Pulse Mode'. Motors will only move")
    print("for 0.2 seconds and then auto-stop. This prevents crashing.\n")

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        print("Error: Could not connect to Arduino.")
        return

    # Joint Mappings (Index in the 6-value array)
    # [YawL, YawR, HipL, KneeL, HipR, KneeR]
    joints = {
        '1': ("Hip Left",  2),
        '2': ("Knee Left", 3),
        '3': ("Hip Right", 4),
        '4': ("Knee Right", 5)
    }

    try:
        while True:
            print("\n--- Select Joint to Calibrate ---")
            print("1: Hip Left")
            print("2: Knee Left")
            print("3: Hip Right")
            print("4: Knee Right")
            print("q: Quit")

            choice = get_user_input("Select (1-4): ")
            if choice == 'q':
                break
            if choice not in joints:
                continue

            joint_name, idx = joints[choice]
            print(f"\n[Calibrating {joint_name}]")
            print("Controls:")
            print("  '+' or 'w' -> Extend (Pulse)")
            print("  '-' or 's' -> Retract (Pulse)")
            print("  'b'        -> Back to Menu")

            while True:
                # 1. Read latest Pot Value
                # Mappings in SDK pot array: [HipL, KneeL, HipR, KneeR]
                # Map our joint selection (2,3,4,5) to pot index (0,1,2,3)
                pot_idx = idx - 2
                current_pot = mcu.latest_pots[pot_idx]

                # 2. Ask for command
                cmd = get_user_input(
                    f"POT VALUE: {current_pot} | Command (+/-): ")

                if cmd == 'b':
                    break

                # 3. Determine Pulse Direction
                # We send a vector with 0.5 (stop) for all, except target
                # Target gets 0.7 (extend) or 0.3 (retract)
                cmds = [0, 0, 0.5, 0.5, 0.5, 0.5]  # Neutral Stance

                if cmd in ['+', 'w']:
                    cmds[idx] = 0.7  # Low speed extend
                    print(f"  -> Extending {joint_name}...")
                elif cmd in ['-', 's']:
                    cmds[idx] = 0.3  # Low speed retract
                    print(f"  -> Retracting {joint_name}...")
                else:
                    continue

                # 4. EXECUTE PULSE (The Safety Feature)
                # Move for only 0.2 seconds, then STOP.
                mcu.send_command(*cmds)
                time.sleep(5)
                mcu.send_command(0, 0, 0.5, 0.5, 0.5, 0.5)  # Hard Stop
                time.sleep(0.1)  # Wait for comms to update pot value

    except KeyboardInterrupt:
        print("\nStopping...")
        mcu.send_command(0, 0, 0.5, 0.5, 0.5, 0.5)
    finally:
        mcu.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Krabby-Uno Linear Actuator Calibration Wizard")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose MCU logs (including safety/runaway alerts)")
    args = parser.parse_args()

    if args.debug:
        # Show underlying MCU SDK debug (POS/POT) plus any safety/runaway alerts
        logging.getLogger("KrabbySDK").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    calibration_wizard()
