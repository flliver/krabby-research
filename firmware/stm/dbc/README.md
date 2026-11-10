# `krabby_leg.dbc`

CAN-FD database for six leg nodes. Numeric message IDs match `../docs/CAN_PROTOCOL.md`, `../docs/FLASH.md`, and `../docs/PROVISIONING.md`.

**Version:** `VERSION` (currently **1.0.0**). **Integrity pin:** `krabby_leg.sha256` (SHA-256 of `krabby_leg.dbc`).

## Regenerate / edit

1. Edit `krabby_leg.dbc` in place (IDs must stay aligned with `../docs/CAN_PROTOCOL.md`).
2. Bump `VERSION` when the on-wire layout changes.
3. Regenerate committed C and refresh the pin:

```bash
# From krabby-research/ with repo .venv (pip install 'cantools>=39.0' once)
make -C firmware stm-dbc
```

Equivalent:

```bash
.venv/bin/python firmware/stm/dbc/regenerate_c.py --python .venv/bin/python
sha256sum firmware/stm/dbc/krabby_leg.dbc | awk '{print $1}' > firmware/stm/dbc/krabby_leg.sha256
```

4. Run `pytest tests/unit/firmware/stm/test_krabby_leg_dbc.py`

## Generated C (`generated/`)

`make -C firmware stm-dbc` writes **`generated/krabby_leg.c`** and **`generated/krabby_leg.h`** via `cantools generate_c_source`. Commit these with DBC edits so MCU builds do not need Python.

Override interpreter: `make -C firmware stm-dbc VENV_PYTHON=/path/to/python`.

## Python (`cantools`)

```bash
.venv/bin/pip install 'cantools>=39.0'
.venv/bin/python -c "import cantools; cantools.database.load_file('firmware/stm/dbc/krabby_leg.dbc')"
```
