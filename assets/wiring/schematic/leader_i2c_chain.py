from dataclasses import dataclass
from pathlib import Path

import schemdraw.elements as elm

from diagram import Diagram
from theme import INK, drawing


WIRE_WIDTH = 1.4
BUNDLE_WIDTH = 2.4
OUTLINE_WIDTH = 1.2
PIN_LENGTH = 0.35
SENSE_COLOR = "#167b83"
MODULE_HEIGHT = 3.6
POWER_Y = -5.7


@dataclass(frozen=True)
class Board:
    x: float
    y: float
    width: float
    height: float

    @property
    def top(self) -> float:
        return self.y + self.height

    @property
    def right(self) -> float:
        return self.x + self.width


def build(svg_path: Path) -> None:
    with drawing(svg_path) as diagram:
        diagram.config(font="Arial", fontsize=10, lw=WIRE_WIDTH)

        def text(at, label, size=10, align="center", color=INK):
            # Label has a default (0.1, 0.1) local center; cancel it for exact placement.
            diagram.add(elm.Label().at(at).theta(0).label(
                label, fontsize=size, halign=align, color=color,
                loc="center", ofst=(-0.1, -0.1), valign="center"))

        def wire(points, *, bundle=False, color=INK):
            for start, end in zip(points, points[1:]):
                diagram.add(elm.Line().at(start).to(end).color(color)
                            .linewidth(BUNDLE_WIDTH if bundle else WIRE_WIDTH).hold())

        def dot(at):
            diagram.add(elm.Dot(radius=0.055).at(at).hold())

        def rectangle(left, bottom, right, top):
            diagram.add(elm.Rect((left, bottom), (right, top),
                                lw=OUTLINE_WIDTH).at((0, 0)).theta(0).hold())

        def board(x, y, width, height, title, detail, *, top_connector=False):
            result = Board(x, y, width, height)
            rectangle(x, y, result.right, result.top)
            inset = 0.5 if top_connector else 0
            text((x + width / 2, result.top - 0.35 - inset), title, 11)
            text((x + width / 2, result.top - 0.8 - inset), detail)
            wire([(x, result.top - 1.15 - inset),
                  (result.right, result.top - 1.15 - inset)])
            return result

        def port(box, side, offset, name="", *, bundle=False):
            # Individual terminals sit on the board boundary; connector groups are outlined.
            # Offsets are measured from the lower/left board edge.
            if side in ("L", "R"):
                x = box.x if side == "L" else box.right
                y = box.y + offset
                direction = -1 if side == "L" else 1
                end = (x + direction * PIN_LENGTH, y)
                label_at = (x - direction * 0.23, y + (0.3 if name == "+" else 0.07))
                align = "left" if side == "L" else "right"
            else:
                x = box.x + offset
                y = box.top if side == "T" else box.y
                direction = 1 if side == "T" else -1
                end = (x, y + direction * PIN_LENGTH)
                label_at = (x, y - 0.5 if side == "T" else y + 0.35)
                align = "center"
            wire([(x, y), end], bundle=bundle)
            if bundle:
                rectangle(x - 0.12, y - 0.16, x + 0.12, y + 0.16)
            else:
                # Opaque fill and higher z-order mask both the outline and wiring.
                diagram.add(elm.Dot(radius=0.075, open=True, zorder=10)
                            .at((x, y)).fill("white").linewidth(WIRE_WIDTH).hold())
            if name:
                text(label_at, name, align=align)
            return end

        def cable(start, end, label=None):
            wire([start, end], bundle=True)
            if label:
                text(((start[0] + end[0]) / 2, start[1] + 0.42), label)

        text((0, 5.4), "Leader — System interconnections", 16, "left")
        text((0, 4.65), "Board interfaces and battery power · not to scale", align="left")

        leader = board(0, -0.4, 4.0, MODULE_HEIGHT + 0.4, "A1 · Leader Mega", "I²C host")
        adapter = board(5.8, 0, 3.6, MODULE_HEIGHT, "A2 · Qwiic adapter", "3.3 V interface")
        for offset, mega_name, adapter_name in [
            (2.05, "3V3", "VCC"), (1.5, "GND", "GND"),
            (0.95, "D20 / SDA", "SDA"), (0.4, "D21 / SCL", "SCL"),
        ]:
            source = port(leader, "R", offset - leader.y, mega_name)
            destination = port(adapter, "L", offset, adapter_name)
            wire([source, destination])

        imu = board(11.5, 0, 3.8, MODULE_HEIGHT, "U1 · LSM6DSO IMU", "I²C 0x6B")
        oled = board(17.4, 0, 4.0, MODULE_HEIGHT, "U2 · SSD1306 OLED", "I²C 0x3D · 128 × 64")
        pack = board(23.5, 0, 4.8, MODULE_HEIGHT, "U3 · Pack INA228", "I²C 0x40")
        midpoint = board(30.4, 0, 4.8, MODULE_HEIGHT, "U4 · Midpoint INA228", "I²C 0x41")
        previous = port(adapter, "R", 1.3, "Qwiic", bundle=True)
        for module in (imu, oled, pack, midpoint):
            incoming = port(module, "L", 1.3, "Qwiic", bundle=True)
            cable(previous, incoming, "Qwiic")
            previous = port(module, "R", 1.3, "Qwiic", bundle=True)

        vin_minus = port(pack, "B", 0.8, "VIN−")
        pack_vbus = port(pack, "B", pack.width / 2, "VBUS")
        vin_plus = port(pack, "B", pack.width - 0.8, "VIN+")
        midpoint_vbus = port(midpoint, "B", midpoint.width / 2, "VBUS")
        text((pack.x + pack.width / 2, 4.1), "U3: SHUNT open · VBUS open")

        shield = board(0, -6.8, 4.0, 3.6, "Krabby-Uno v0.2", "Shield", top_connector=True)
        mega_headers = port(leader, "B", leader.width / 2, "Shield headers", bundle=True)
        shield_headers = port(shield, "T", shield.width / 2, "Mega headers", bundle=True)
        cable(mega_headers, shield_headers)
        text((mega_headers[0] + 0.4, -1.7), "Stacking headers", align="left")

        motor_power = []
        for header, role, bottom, shield_offset in [
            ("J1", "FL", -4.1, 1.3), ("J2", "FR", -9.8, 0.5),
        ]:
            motor = board(7.3, bottom, 4.7, 2.7, f"{role} MCU board", "Motor / actuator control")
            control = port(motor, "L", 0.8, "2×10", bundle=True)
            motor_power.append(port(motor, "R", 0.5, "Power +"))
            source = port(shield, "R", shield_offset, header, bundle=True)
            elbow_x = (shield.right + motor.x) / 2
            wire([source, (elbow_x, source[1]), (elbow_x, control[1]), control], bundle=True)
            text(((elbow_x + control[0]) / 2, control[1] + 0.65), "20-pin\nribbon")

        # Battery polarity and order are unchanged: positive on the left.
        def battery(positive_x, name):
            positive = (positive_x, POWER_Y)
            symbol = diagram.add(elm.Battery().at(positive).right().length(4.0).hold())
            negative = tuple(symbol.end)
            text((positive_x + 2.0, POWER_Y + 1.05), f"Battery {name} · 12 V", 11)
            text((positive_x + 1.25, POWER_Y + 0.4), "+")
            text((positive_x + 2.75, POWER_Y + 0.4), "−")
            return positive, negative

        battery_b_pos, battery_b_neg = battery(pack_vbus[0], "B")
        battery_a_pos, battery_a_neg = battery(midpoint_vbus[0], "A")
        wire([battery_b_neg, battery_a_pos])
        wire([pack_vbus, battery_b_pos])
        wire([midpoint_vbus, battery_a_pos])
        dot(battery_b_pos)
        dot(battery_a_pos)

        fuse = diagram.add(elm.Fuse().at(battery_b_pos).left().length(2.6)
                           .label("F1 · 150 A", fontsize=10).hold())
        shunt = diagram.add(elm.Resistor().at(fuse.end).left().length(2.6)
                            .label("Shunt", fontsize=10).hold())
        wire([vin_minus, (vin_minus[0], -1.5), (shunt.end.x, -1.5), shunt.end], color=SENSE_COLOR)
        # This is the only crossing: VIN+ bridges VBUS without connecting.
        crossing_x = pack_vbus[0]
        bridge_y = -2.5
        wire([vin_plus, (vin_plus[0], bridge_y), (crossing_x + 0.22, bridge_y)], color=SENSE_COLOR)
        diagram.add(elm.Arc2(k=0.8).at((crossing_x + 0.22, bridge_y))
                    .to((crossing_x - 0.22, bridge_y)).color(SENSE_COLOR)
                    .linewidth(WIRE_WIDTH).hold())
        wire([(crossing_x - 0.22, bridge_y), (shunt.start.x, bridge_y), shunt.start], color=SENSE_COLOR)
        dot(shunt.start)
        dot(shunt.end)

        octopus = board(shunt.end.x - 6.5, POWER_Y - 1.6, 4.7, 3.6,
                        "Octopus", "Power distribution block")
        incoming = port(octopus, "R", 1.6, "+")
        outputs = [port(octopus, "L", offset, "+") for offset in (1.6, 0.5)]
        wire([shunt.end, incoming])
        # Common positive conductor inside the distribution assembly.
        bus_x = octopus.x + octopus.width / 2
        wire([(octopus.right, incoming[1]), (bus_x, incoming[1]), (bus_x, outputs[1][1])])
        for output, destination in zip(outputs, motor_power):
            wire([(octopus.x, output[1]), (bus_x, output[1])])
            dot((bus_x, output[1]))
            route_x = (octopus.x + destination[0]) / 2
            wire([output, (route_x, output[1]), (route_x, destination[1]), destination])

        text((battery_b_pos[0], POWER_Y - 1.35), "Pack + · 24 V nominal", align="right")
        text((battery_a_pos[0], POWER_Y - 1.35), "Midpoint")
        text((battery_a_neg[0], POWER_Y - 1.35), "Pack −")

        text((0, -10.8), "CAUTION: Ensure +3V3 is never accidentally connected to Mega 5V.", align="left")


DIAGRAM = Diagram(
    name=Path(__file__).stem,
    title="Krabby M16 — Leader system interconnections",
    hint="Leader Mega and shield, Qwiic sensor chain, and battery power distribution. "
         "Two 12 V batteries in series; direct VBUS taps; pack-positive feed through "
         "the 150 A fuse and shunt to the Octopus and FL / FR MCU boards. "
         "Pack VIN+ senses the fuse side; VIN− senses the outgoing side.",
    build=build,
)
