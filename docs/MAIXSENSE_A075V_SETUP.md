# MaixSense-A075V setup (Jetson)

RGB-D over **USB RNDIS + HTTP** (no ZED SDK). Official hardware docs: [MaixSense-A075V – Sipeed Wiki](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/maixsense-a075v.html).

**HAL catalog (repo defaults):**

| Side | Catalog id | Module IP (`maixsense_host`) |
|------|------------|------------------------------|
| Right | `side_right_rgbd` | **`192.168.233.1:80`** (factory) |
| Left | `side_left_rgbd` | **`192.168.233.2:80`** (set on module over SSH) |

Also used with **`front_rgbd`** when **`camera_driver="maixsense_a075v"`**. Endpoints are literals in **`JETSON_SENSOR_CATALOG`** (`hal/server/jetson/sensor_backend_jetson.py`). Docker: **`--network host`**; no **`-e`** for MaixSense IPs.

---

## Single module (USB link)

1. **USB** — Often **`0525:a4a2`** (RNDIS). Module default IP **`192.168.233.1`** on **`usb0`**.
2. **Jetson** — Module **`udhcpd`** usually assigns **`192.168.233.100/24`** on the RNDIS link (**`enx…`** or **`usb0` / `usb1`**, depending on udev); no manual **`ip addr`** for one module. Confirm with `ip -br link`.
3. **Reachability** — `ip route get 192.168.233.1`; `ping -c 3 192.168.233.1`; `curl -sS -o /dev/null -w '%{http_code}\n' http://192.168.233.1/` (expect **`200`**).
4. **Web UI** — `http://192.168.233.1` (~15 s after power-on). Remote: `ssh -N -L 8080:192.168.233.1:80 USER@JETSON` → `http://127.0.0.1:8080`.
5. **SSH to module** — **`root` / `root`**; OpenSSH needs **`+ssh-rsa`**:

   ```bash
   ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.233.1
   ```

6. **Drivers (troubleshooting only)** — [Sipeed install / driver notes](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/install_drivers.html) if USB RNDIS does not appear.

**Not supported for HAL:** UART instead of USB-C — the Jetson driver is HTTP over the RNDIS link only.

---

## Two side modules

Both stay on **`192.168.233.0/24`**. Only the **left** module changes IP (**`.1` → `.2`**). The **right** stays at factory **`.1`**.

Configure the **left** with **only that module** plugged in (both ship at **`.1`**; you cannot target one module over SSH while both share the address).

### Step 1 — Left module only (right unplugged)

Unplug the right MaixSense. Connect the **left** only. Wait ~15 s.

**On Jetson:**

```bash
lsusb | grep -c '0525:a4a2'    # must print 1
curl -sS -o /dev/null -w '%{http_code}\n' http://192.168.233.1/
ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.233.1
```

**On the module** — identify interface (stock A075V: **`usb0`**, not **`lo`**):

```bash
ls /sys/class/net/
ifconfig -a
grep -rn '192.168.233.1' /etc /root 2>/dev/null
ifconfig usb0 | grep -E 'inet addr:|inet '
```

Persist IP change in **`/etc/init.d/rc.preboot`** (while still SSH'd at **`.233.1`**):

```bash
CONF=/etc/init.d/rc.preboot
cp -a "$CONF" "$CONF.bak.$(date +%Y%m%d)"
sed -i 's/ifconfig usb0 192.168.233.1/ifconfig usb0 192.168.233.2/' "$CONF"
grep ifconfig "$CONF"
```

Apply address (SSH will drop):

```bash
ifconfig usb0 192.168.233.2 netmask 255.255.255.0 up
```

**On Jetson** — reconnect:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://192.168.233.2/
ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.233.2
```

**On module** — `ifconfig usb0` then `reboot`. After ~15 s on Jetson: `curl` to **`.2`** should return **`200`**.

### Step 2 — Connect both modules

Plug **right** and **left** into **separate USB ports**. Wait ~15 s. Right stays at **`.1`**; no module-side work.

### Step 3 — Persist `/32` routes (NetworkManager dispatcher)

Stock Jetson Ubuntu 22.04 images use NetworkManager for the USB RNDIS links; confirm before installing:

```bash
systemctl is-active NetworkManager    # expect: active
ip -br link | awk '/^(enx|usb[0-9])/ { print }'   # expect enx… and/or usb0, usb1
```

Host iface names vary: some images show **`enx…`** (MAC-based), others **`usb0` / `usb1`**. The dispatcher must match **both**; an `enx`-only filter is a silent no-op when links are named `usb*`.

With both modules on the same **`/24`**, the kernel may route **`.1`** and **`.2`** through one RNDIS iface after reboot. Install a dispatcher hook: on each RNDIS **`up`**, discover which link owns each IP (ping with **`-I`**) and **`ip route replace …/32`**. Partial mapping is OK — the first **`up`** may only map one side; the second **`up`** completes both.

```bash
sudo tee /etc/NetworkManager/dispatcher.d/99-maixsense-routes > /dev/null <<'EOF'
#!/bin/bash
IFACE="$1"
ACTION="$2"

[[ "$ACTION" == "up" ]] || exit 0
[[ "$IFACE" =~ ^(enx|usb[0-9]+) ]] || exit 0

RIGHT_IP=192.168.233.1
LEFT_IP=192.168.233.2
BOOT_WAIT_SEC=15

sleep "$BOOT_WAIT_SEC"

right_usb=""
left_usb=""
for dev in $(ip -br link | awk '/^(enx|usb[0-9]+)/ && $2 == "UP" { print $1 }'); do
  ping -c 1 -W 1 -I "$dev" "$RIGHT_IP" >/dev/null 2>&1 && right_usb="$dev"
  ping -c 1 -W 1 -I "$dev" "$LEFT_IP" >/dev/null 2>&1 && left_usb="$dev"
done

[[ -n "$right_usb" ]] && ip route replace "${RIGHT_IP}/32" dev "$right_usb"
[[ -n "$left_usb" ]] && ip route replace "${LEFT_IP}/32" dev "$left_usb"

exit 0
EOF
sudo chmod 755 /etc/NetworkManager/dispatcher.d/99-maixsense-routes
```

If an older `enx`-only copy of this script is already installed, replace it with the version above (or the hook will never run on `usb0`/`usb1` hosts).

Replug USB or `sudo nmcli device reapply <iface>` for each RNDIS link (`enx…` or `usb0`/`usb1`).

**Immediate (non-persistent) check** — after discovering which iface owns which IP with `ping -I`:

```bash
sudo ip route replace 192.168.233.1/32 dev <right_iface>
sudo ip route replace 192.168.233.2/32 dev <left_iface>
```

### Step 4 — Verify both modules

```bash
ip route get 192.168.233.1
ip route get 192.168.233.2
curl -sS -o /dev/null -w 'right %{http_code}\n' http://192.168.233.1/
curl -sS -o /dev/null -w 'left %{http_code}\n' http://192.168.233.2/
```

Expect different **`dev`** per destination (`enx…` or `usbN`); both HTTP **`200`**. Reboot and repeat before starting HAL.

### Step 5 — Verify catalog

Confirm **`maixsense_host`** in `hal/server/jetson/sensor_backend_jetson.py` matches **`.1`** / **`.2`** above.
