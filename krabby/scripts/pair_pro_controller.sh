#!/bin/bash
# Pairs Nintendo Switch Pro Controller and persists the link key despite
# store_hint=0, enabling auto-reconnect across bluetooth restarts.
# Works on a clean Jetson with no prior pairing — discovers controller by name.
BTMON_LOG=$(mktemp /tmp/btmon.XXXXXX)
BTMON_PID=""
SCAN_PID=""
INTERRUPTED=0

bt() { bluetoothctl -- "$@" > /dev/null 2>&1 || true; }

cleanup() {
    [ $INTERRUPTED -eq 1 ] && echo "" && echo "[interrupted] cleaning up..."
    [ -n "$SCAN_PID" ] && kill "$SCAN_PID" 2>/dev/null || true
    bt scan off
    bt discoverable off
    bt pairable off
    [ -n "$BTMON_PID" ] && sudo kill "$BTMON_PID" 2>/dev/null || true
    rm -f "$BTMON_LOG"
}

handle_sigint() {
    INTERRUPTED=1
    cleanup
    exit 130
}

trap handle_sigint INT TERM
trap cleanup EXIT

echo "[1] Getting adapter MAC..."
ADAPTER_MAC=$(bluetoothctl show | awk '/Controller/{print $2; exit}')
echo "    Adapter: $ADAPTER_MAC"

echo "[2] Starting btmon to capture link key..."
sudo btmon --no-pager > "$BTMON_LOG" 2>&1 &
BTMON_PID=$!
sleep 0.5

echo "[3] Making host discoverable..."
bt discoverable on
bt pairable on

echo "[4] Scanning for device named \"Pro Controller\"…"
echo "    Official Pro: hold Sync (pinhole by USB-C) until ALL 4 LEDs flash rapidly (Home ≠ Sync)."
echo "    Third-party / no Sync: prefer USB, or bluetoothctl (see CONNECT_PRO_CONTROLLER.md)."
bt scan on &
SCAN_PID=$!

MAC=""
FOUND_TOO_QUICKLY=0
for i in $(seq 1 25); do
    MAC=$(bluetoothctl -- devices | awk '/Pro Controller/{print $2; exit}')
    if [ -n "$MAC" ]; then
        echo "    Found Pro Controller at $MAC after ${i}s"
        if [ "$i" -lt 3 ]; then
            FOUND_TOO_QUICKLY=1
        fi
        break
    fi
    sleep 1
done
kill $SCAN_PID 2>/dev/null; SCAN_PID=""
bt scan off

if [ -z "$MAC" ]; then
    echo "[err] No device named \"Pro Controller\" found."
    echo "      Official Pro: power off (Home ~10s) → Sync until 4 LEDs flash → re-run pair-pro."
    echo "      Third-party: name may not be \"Pro Controller\" — use USB or bluetoothctl scan"
    echo "      (CONNECT_PRO_CONTROLLER.md §A / §C). Do not chase hid_nintendo for a missing name."
    exit 1
fi

if [ "$FOUND_TOO_QUICKLY" -eq 1 ]; then
    # Warn only — a pad already in Sync (or a clone advertising) can also appear in <3s.
    echo "[warn] Found too quickly — often a Home/cache reconnect, not Sync pairing mode."
    echo "       Home ≠ Sync. If later Paired: no, that is an incomplete bond (not hid_nintendo)."
    echo "       Official Pro: power off → bluetoothctl remove $MAC → Sync → re-run pair-pro."
    echo "       Third-party / frequent bench: use USB (ls /dev/input/js*) or bluetoothctl —"
    echo "       see CONNECT_PRO_CONTROLLER.md. Continuing this attempt anyway…"
fi

echo "[5] Removing any stale entry for $MAC..."
bt remove "$MAC"
sleep 0.5

echo "[6] Trusting + pairing (AuthenticationCanceled is OK)..."
bt trust "$MAC"
bt pair "$MAC"

echo "[7] Waiting up to 30s for connection..."
CONNECTED=0
for i in $(seq 1 30); do
    if timeout 5 bluetoothctl -- info "$MAC" | grep -q "Connected: yes"; then
        echo "    Connected after ${i}s"
        CONNECTED=1
        # Set player 1 LED; sysfs node appears shortly after connection
        sleep 0.5
        LED_BASE=$(ls /sys/class/leds/ 2>/dev/null | grep "057E:2009.*:player1" | head -1 | sed 's/:player1//')
        if [ -n "$LED_BASE" ]; then
            echo 1 | sudo tee "/sys/class/leds/${LED_BASE}:player1/brightness" > /dev/null
            for p in 2 3 4; do
                echo 0 | sudo tee "/sys/class/leds/${LED_BASE}:player${p}/brightness" > /dev/null
            done
            echo "    Player LED set to 1"
        fi
        break
    fi
    [ $((i % 5)) -eq 0 ] && timeout 8 bluetoothctl -- connect "$MAC" > /dev/null 2>&1 || true
    sleep 1
done

if [ $CONNECTED -eq 0 ]; then
    echo "[err] Failed to connect within 30s — try running the script again"
    exit 1
fi

echo "[8] Waiting for pairing to complete..."
PAIRED=0
for i in $(seq 1 15); do
    if timeout 5 bluetoothctl -- info "$MAC" | grep -q "Paired: yes"; then
        echo "    Paired after additional ${i}s"
        PAIRED=1
        break
    fi
    sleep 1
done
if [ "$PAIRED" -eq 0 ]; then
    echo "[warn] Still Paired: no after wait — will report failure at the end (not hid_nintendo)."
fi
sleep 0.5
kill "$BTMON_PID" 2>/dev/null || true; BTMON_PID=""
echo "[9] Extracting link key from btmon log..."

LINK_KEY_LINE=$(grep -A3 "Link Key Notification" "$BTMON_LOG" | grep "Link key:" | tail -1)
if [ -z "$LINK_KEY_LINE" ]; then
    echo "[warn] Link key not found in btmon log — reconnect will require re-pairing"
else
    KEY_HEX=$(echo "$LINK_KEY_LINE" | sed 's/.*Link key: //' | tr -d ' \n' | tr 'a-f' 'A-F')
    KEY_TYPE_LINE=$(grep -A4 "Link Key Notification" "$BTMON_LOG" | grep "Key type:" | tail -1)
    KEY_TYPE_HEX=$(echo "$KEY_TYPE_LINE" | grep -oP '0x\K[0-9a-fA-F]+' | head -1)
    KEY_TYPE_DEC=$((16#${KEY_TYPE_HEX:-04}))

    echo "    Key: ${KEY_HEX:0:8}... Type: $KEY_TYPE_DEC"

    INFO_PATH="/var/lib/bluetooth/$ADAPTER_MAC/$MAC/info"
    echo "[10] Writing link key to $INFO_PATH..."
    sudo mkdir -p "$(dirname "$INFO_PATH")"
    if sudo test -f "$INFO_PATH"; then
        sudo sed -i '/^\[LinkKey\]/,/^[[:space:]]*$/d' "$INFO_PATH"
    fi
    printf '[LinkKey]\nKey=%s\nType=%d\nPINLength=0\n\n' \
        "$KEY_HEX" "$KEY_TYPE_DEC" | sudo tee -a "$INFO_PATH" > /dev/null
    echo "    Persisted. Controller will auto-reconnect after bluetooth restarts."
fi

echo ""
echo "[done] Final status:"
INFO=$(timeout 5 bluetoothctl -- info "$MAC" 2>/dev/null || true)
echo "$INFO" | grep -E '(Name|Connected|Paired|Trusted)' || true
PAIRED=0
echo "$INFO" | grep -q "Paired: yes" && PAIRED=1

JS=""
for i in $(seq 1 5); do
    JS=$(ls /dev/input/js* 2>/dev/null | tr '\n' ' ')
    [ -n "$JS" ] && break
    sleep 1
done

if [ "$PAIRED" -eq 0 ]; then
    echo "[err] Pairing incomplete (Paired: no) — the Bluetooth bond did not finish."
    echo "      Do not chase hid_nintendo for this."
    echo "      Official Nintendo Pro:"
    echo "        1. Power off (Home ~10s)  2. bluetoothctl remove $MAC"
    echo "        3. Sync (pinhole by USB-C) until ALL 4 LEDs flash  4. re-run pair-pro"
    echo "      Third-party (no Sync / clones like XB-324): BT often never sticks — prefer USB:"
    echo "        plug data cable → ls /dev/input/js* → krabby run --gamepad-only"
    echo "      Or manual BT: bluetoothctl pair/trust/connect (long-Home or maker pair button)."
    echo "      Full guide: controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md"
    echo "      Success looks like: Paired: yes, Connected: yes, and /dev/input/js0."
    exit 1
fi

if [ -n "$JS" ]; then
    echo "js device: $JS"
else
    if lsmod | grep -q hid_nintendo; then
        echo "[warn] Paired: yes but no /dev/input/js* yet — wait a few seconds or press Home to reconnect."
        echo "       hid_nintendo is already loaded; check udev (krabby install) if js* never appears."
    else
        echo "[warn] Paired: yes but no js device — hid_nintendo is not loaded."
        echo "       Try: sudo modprobe hid_nintendo"
        echo "       Or re-run host setup after a kernel update:"
        echo "         sudo -E env PATH=\"\$PATH\" \"\$(which krabby)\" install"
    fi
    exit 1
fi
