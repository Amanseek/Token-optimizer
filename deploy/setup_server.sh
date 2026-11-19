#!/usr/bin/env bash
# Always-on Tokenwise on a fresh Ubuntu server (22.04 or 24.04), with HTTPS on your domain.
#
#   curl -fsSL https://raw.githubusercontent.com/Amanseek/Token-optimizer/main/deploy/setup_server.sh -o setup.sh
#   sudo bash setup.sh api.yourdomain.com
#
# Before running: point the domain's DNS "A" record at this server's public IP.
# Run it again any time to update Tokenwise to the latest version (keys and data are kept).
#
# What it sets up:
#   - Tokenwise in /opt/tokenwise, running as its own user, restarted if it crashes or the server reboots
#   - Caddy in front: free HTTPS certificate for your domain, renewed automatically
#   - firewall: only SSH (22) and web (80, 443) open; Tokenwise itself listens on 127.0.0.1 only
#   - keys in /etc/tokenwise/env (root only), database in /var/lib/tokenwise
#   - `tw-admin` command for client accounts, e.g.  sudo tw-admin list
set -euo pipefail

DOMAIN="${1:-}"
REPO=${REPO:-https://github.com/Amanseek/Token-optimizer}
APP=/opt/tokenwise
DATA=/var/lib/tokenwise
CONF=/etc/tokenwise
[ "$(id -u)" = 0 ] || { echo "Run with sudo (as root)."; exit 1; }
[ -n "$DOMAIN" ] || { echo "Usage: sudo bash setup.sh api.yourdomain.com"; exit 1; }

echo "== 1/6 check the domain points here"
apt-get update -qq && apt-get install -y -qq curl dnsutils > /dev/null
MY_IP=$(curl -fsS4 https://api.ipify.org || true)
DNS_IP=$(dig +short A "$DOMAIN" @1.1.1.1 | tail -1)
echo "this server: ${MY_IP:-unknown} | $DOMAIN points to: ${DNS_IP:-nothing yet}"
if [ -z "$DNS_IP" ] || [ "$DNS_IP" != "$MY_IP" ]; then
  echo "The domain doesn't point to this server yet, so HTTPS can't be set up."
  echo "In your DNS (e.g. Hostinger): add an A record, name = the part before the main domain"
  echo "(for api.prankundali.com: 'api'), value = $MY_IP. Wait a few minutes and run this again."
  exit 1
fi

echo "== 2/6 packages"
apt-get install -y -qq git python3 python3-venv debian-keyring debian-archive-keyring apt-transport-https gnupg ufw > /dev/null
if ! command -v caddy > /dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy > /dev/null
fi

echo "== 3/6 Tokenwise"
id tokenwise > /dev/null 2>&1 || useradd --system --home "$DATA" --shell /usr/sbin/nologin tokenwise
mkdir -p "$DATA" && chown tokenwise:tokenwise "$DATA" && chmod 700 "$DATA"
if [ -d "$APP/.git" ]; then git -C "$APP" pull -q; else git clone -q "$REPO" "$APP"; fi
[ -x "$APP/venv/bin/python" ] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q -r "$APP/requirements.txt"

echo "== 4/6 keys"
mkdir -p "$CONF" && chmod 700 "$CONF"
if [ ! -f "$CONF/env" ]; then  # created once; kept on updates
  umask 077
  cat > "$CONF/env" <<EOF
TOKENWISE_SECRET=$(cd "$APP" && "$APP/venv/bin/python" -m tokenwise.admin new-secret 2>/dev/null)
TOKENWISE_ADMIN_TOKEN=$("$APP/venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(24))')
TOKENWISE_DB=$DATA/tokenwise.db
TOKENWISE_REQUIRE_CLIENT_KEY=1
TOKENWISE_OPENAI_PRICES='{"gemini-2.5-flash": [0.30, 2.50], "gpt-5-mini": [0.25, 2.00]}'
TOKENWISE_INR_PER_USD=95
EOF
fi
cat > /usr/local/bin/tw-admin <<EOF
#!/usr/bin/env bash
# Client accounts: sudo tw-admin list | create NAME ... | fallback ... | shadow ... (see tokenwise/admin.py)
set -a; . $CONF/env; set +a
cd $APP && exec sudo -E -u tokenwise $APP/venv/bin/python -m tokenwise.admin "\$@"
EOF
chmod 750 /usr/local/bin/tw-admin

echo "== 5/6 services"
cat > /etc/systemd/system/tokenwise.service <<EOF
[Unit]
Description=Tokenwise AI gateway
After=network-online.target
Wants=network-online.target

[Service]
User=tokenwise
WorkingDirectory=$APP
EnvironmentFile=$CONF/env
ExecStart=$APP/venv/bin/python -m uvicorn tokenwise.main:app --host 127.0.0.1 --port 8787 --proxy-headers
Restart=always
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$DATA
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
	encode gzip
	reverse_proxy 127.0.0.1:8787 {
		flush_interval -1
	}
}
EOF
systemctl daemon-reload
systemctl enable -q --now tokenwise
systemctl restart tokenwise
systemctl enable -q caddy && systemctl reload-or-restart caddy

echo "== 6/6 firewall"
ufw allow OpenSSH > /dev/null && ufw allow 80/tcp > /dev/null && ufw allow 443/tcp > /dev/null
ufw --force enable > /dev/null

echo "checking https://$DOMAIN (the first certificate can take a minute)..."
for _ in $(seq 30); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://$DOMAIN/dashboard" || true)
  [ "$code" = 403 ] || [ "$code" = 200 ] && break
  sleep 4
done
if [ "${code:-}" != 403 ] && [ "${code:-}" != 200 ]; then
  echo "HTTPS isn't answering yet. Check: systemctl status tokenwise caddy  |  journalctl -u caddy -n 30"
  exit 1
fi
cat <<EOF

READY. Tokenwise is running at https://$DOMAIN (locked: dashboard needs your admin token, API needs a client key)
  Dashboard link (private):  sudo bash -c '. $CONF/env; echo https://$DOMAIN/dashboard?token=\$TOKENWISE_ADMIN_TOKEN'
  Apps (OpenAI format):      https://$DOMAIN/openai/v1      with a client's tw- key
  Client accounts:           sudo tw-admin list
  Logs:                      journalctl -u tokenwise -f
  Update later:              sudo bash setup.sh $DOMAIN
EOF
