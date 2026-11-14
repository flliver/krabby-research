# `krabby.dbc`

Leg-board CAN bus schema: 19 CAN-FD messages, 11-bit IDs (`VFrameFormat` = StandardCAN_FD).

**Integrity pin:** `krabby.sha256`

## Regenerate

```bash
make -C firmware stm-dbc
pytest tests/unit/firmware/stm/test_krabby_dbc.py
```

Or:

```bash
.venv/bin/python firmware/stm/dbc/regenerate_c.py
sha256sum firmware/stm/dbc/krabby.dbc | awk '{print $1}' > firmware/stm/dbc/krabby.sha256
```

Generated **`generated/krabby.c`** and **`generated/krabby.h`** are committed so MCU builds need no Python.