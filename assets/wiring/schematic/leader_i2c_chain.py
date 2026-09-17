from pathlib import Path

import schemdraw
import schemdraw.elements as elm

from diagram import Diagram
from theme import MUTED, add_title, drawing


NETS = (
    ("+3V3", "VCC", "4/4"),
    ("GND", "GND", "3/4"),
    ("I2C_SDA", "SDA", "2/4"),
    ("I2C_SCL", "SCL", "1/4"),
)
MEGA_PIN_LABELS = ("3V3", "GND", "D20 / SDA", "D21 / SCL")
ADAPTER_PIN_LABELS = ("VCC", "GND", "SDA", "SCL")


def module_pins(
    side: str,
    labels: tuple[str, ...],
    suffix: str = "",
) -> list[elm.IcPin]:
    return [
        elm.IcPin(
            name=label,
            side=side,
            slot=slot,
            anchorname=f"{anchor}{suffix}",
            lblsize=11,
        )
        for label, (_, anchor, slot) in zip(labels, NETS)
    ]


def connect_nets(
    diagram: schemdraw.Drawing,
    source: elm.Ic,
    source_suffix: str,
    destination: elm.Ic,
    destination_suffix: str,
) -> None:
    for _, anchor, _ in NETS:
        diagram.add(
            elm.Wire("-")
            .at(getattr(source, f"{anchor}{source_suffix}"))
            .to(getattr(destination, f"{anchor}{destination_suffix}"))
            .hold()
        )


def build(svg_path: Path) -> None:
    with drawing(svg_path) as diagram:
        add_title(diagram, "LEADER I²C Chain")

        leader = diagram.add(
            elm.Ic(
                size=(4.8, 5.6),
                pins=module_pins("R", MEGA_PIN_LABELS),
            )
            .side("R", spacing=1.0)
            .at((0, 0))
            .theta(0)
            .label("A1\n\nI²C host\n(Leader Mega)")
        )

        adapter = diagram.add(
            elm.Ic(
                size=(4.0, 4.0),
                pins=module_pins("L", ADAPTER_PIN_LABELS, "_IN")
                + [
                    elm.IcPin(
                        name="QWIIC",
                        side="R",
                        slot="1/1",
                        anchorname="QWIIC",
                        lblsize=11,
                    )
                ],
            )
            .at((6.8, 0.8))
            .theta(0)
            .label("A2\n\nQWIIC\nADAPTER")
        )

        imu = diagram.add(
            elm.Ic(
                size=(5.6, 5.6),
                pins=[
                    elm.IcPin(
                        name="QWIIC",
                        side="L",
                        slot="1/1",
                        anchorname="QWIIC_IN",
                        lblsize=11,
                    ),
                    elm.IcPin(
                        name="QWIIC",
                        side="R",
                        slot="1/1",
                        anchorname="QWIIC_OUT",
                        lblsize=11,
                    )
                ],
            )
            .at((13.4, 0))
            .theta(0)
            .label("U1\n\nLSM6DSO IMU\n0x6B")
        )

        oled = diagram.add(
            elm.Ic(
                size=(5.6, 5.6),
                pins=[
                    elm.IcPin(
                        name="QWIIC",
                        side="L",
                        slot="1/1",
                        anchorname="QWIIC",
                        lblsize=11,
                    )
                ],
            )
            .at((21.6, 0))
            .theta(0)
            .label("U2\n\nSSD1306 OLED\n0x3D\n128 × 64")
        )

        connect_nets(diagram, leader, "", adapter, "_IN")
        diagram.add(elm.BusLine().at(adapter.QWIIC).to(imu.QWIIC_IN).hold())
        diagram.add(elm.BusLine().at(imu.QWIIC_OUT).to(oled.QWIIC).hold())

        diagram.add(
            elm.Label()
            .at((0, -2.4))
            .label(
                "CAUTION: Ensure +3V3 is never accidentally connected to Mega 5V.",
                halign="left",
            )
        )


DIAGRAM = Diagram(
    name=Path(__file__).stem,
    title="Krabby M16 — Leader IMU / OLED / I2C",
    hint="Leader Mega → Qwiic adapter → LSM6DSO IMU → SSD1306 OLED.",
    build=build,
)
