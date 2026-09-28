"""Entry point for the Jetson HAL server (``krabby-hal-server-jetson``).

Routes on --control-source to one of two implementations so gamepad mode never
imports the policy (torch) or teleop stacks:

- gamepad             -> hal.server.jetson.main_gamepad (HAL over TCP for a krabby-uno client)
- inference / portal  -> hal.server.jetson.main_model (in-process policy and/or WebRTC teleop)

All other arguments pass through unchanged to the selected implementation.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-source", default="portal", choices=["portal", "inference", "gamepad"])
    args, _ = parser.parse_known_args()

    if args.control_source == "gamepad":
        from hal.server.jetson.main_gamepad import main as run
    else:
        from hal.server.jetson.main_model import main as run
    run()


if __name__ == "__main__":
    main()
