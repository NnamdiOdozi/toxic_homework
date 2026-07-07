#!/usr/bin/env bash
# YET TO USE THIS SCRIPT AND IT DEFINITELY NEEDS TO BE TESTED. USE AT YOUR OWN RISK.
# nebius_provision.sh
#
# Creates and starts a disposable Nebius H100 VM with:
#   - 1 x NVIDIA H100 GPU
#   - 16 vCPUs and 80 GiB RAM
#   - 200 GiB managed SSD boot disk
#   - Ubuntu 24.04 with CUDA 13.0
#   - Auto-assigned static public IP address
#   - SSH user: nnamd
#   - SSH key: ~/.ssh/nebius_vm_key
#   - Service account: mlflow-sa
#
# Run:
#   chmod +x nebius_provision.sh
#   ./nebius_provision.sh
#
# Skip the confirmation prompt:
#   ./nebius_provision.sh --yes
#
# Optional overrides:
#   VM_NAME=my-h100 SUBNET_ID=vpcsubnet-... ./nebius_provision.sh
#
# IMPORTANT:
#   The private SSH key remains on this laptop. Only the public key is
#   supplied to the VM.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_VM_NAME="$(basename "${PROJECT_DIR:-$(pwd)}" | tr '_' '-')-h100"
VM_NAME="${VM_NAME:-$DEFAULT_VM_NAME}"
VM_HOST_ALIAS="${VM_HOST_ALIAS:-nebius-vm}"
VM_USER="${VM_USER:-nnamd}"

PRIVATE_KEY="${PRIVATE_KEY:-$HOME/.ssh/nebius_vm_key}"
PUBLIC_KEY="${PUBLIC_KEY:-${PRIVATE_KEY}.pub}"

EXPECTED_PROJECT_NAME="${EXPECTED_PROJECT_NAME:-default-project-eu-north1}"
TENANT_ID="${TENANT_ID:-tenant-e00vgv8t9as7zvq2xy}"
PROJECT_ID="${PROJECT_ID:-}"

# ---------------------------------------------------------------------------
# Platform & preset — swap these 3 lines to change VM type
# ---------------------------------------------------------------------------
#
# Available platforms (as of June 2026):
#
#   Use Case       | Platform        | Example Preset            | Region
#   -------------- | --------------- | ------------------------- | -------------------------
#   H100 (current) | gpu-h100-sxm    | 1gpu-16vcpu-200gb         | eu-north1
#   H200           | gpu-h200-sxm    | 1gpu-16vcpu-200gb         | eu-north1, eu-west1, us-central1
#   L40S (cheaper) | gpu-l40s-a      | 1gpu-8vcpu-32gb           | eu-north1
#   L40S AMD       | gpu-l40s-d      | 1gpu-16vcpu-96gb          | eu-north1
#   CPU AMD        | cpu-d3          | 4vcpu-16gb to 256vcpu-1024gb | all regions
#   CPU Intel      | cpu-e2          | 2vcpu-8gb to 80vcpu-320gb | eu-north1
#
#   Multi-GPU: use 8gpu-128vcpu-1600gb preset (H100/H200) for 8x GPU nodes
#
# To switch to CPU:
#   PLATFORM="cpu-d3"  PRESET="4vcpu-16gb"  IMAGE_FAMILY="ubuntu24.04"
#
# To switch to H200:
#   PLATFORM="gpu-h200-sxm"  (same preset works)
#
# To switch to 8x H100:
#   PRESET="8gpu-128vcpu-1600gb"  (same platform)
#
# Override from command line without editing this file:
#   PLATFORM=cpu-d3 PRESET=4vcpu-16gb IMAGE_FAMILY=ubuntu24.04 ./scripts/nebius_provision.sh
# ---------------------------------------------------------------------------

PLATFORM="${PLATFORM:-gpu-h100-sxm}"
PRESET="${PRESET:-1gpu-16vcpu-200gb}"

BOOT_DISK_NAME="${BOOT_DISK_NAME:-${VM_NAME}-boot}"
BOOT_DISK_SIZE_GIB="${BOOT_DISK_SIZE_GIB:-500}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-network_ssd}"
# GPU images use cuda suffix; CPU images drop it: ubuntu24.04
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}"

SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-mlflow-sa}"

STATE_FILE="${STATE_FILE:-$HOME/.nebius-gpu-state.env}"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"

SKIP_CONFIRMATION=false
VM_ID=""
PUBLIC_IP=""

# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage:
  ./nebius_provision.sh
  ./nebius_provision.sh --yes
  ./nebius_provision.sh --help

Options:
  --yes   Create the VM without asking for confirmation.
  --help  Show this help message.

Common environment-variable overrides:
  VM_NAME
  VM_HOST_ALIAS
  VM_USER
  PRIVATE_KEY
  PUBLIC_KEY
  PROJECT_ID
  EXPECTED_PROJECT_NAME
  SUBNET_ID
  SERVICE_ACCOUNT_NAME
  BOOT_DISK_SIZE_GIB
  STATE_FILE
USAGE
}

fail() {
    echo
    echo "ERROR: $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?

    echo >&2
    echo "============================================================" >&2
    echo "PROVISIONING DID NOT COMPLETE CLEANLY" >&2
    echo "============================================================" >&2

    if [[ -n "${VM_ID:-}" ]]; then
        echo "A VM may already exist and may be accruing charges." >&2
        echo "VM ID: $VM_ID" >&2
        echo >&2
        echo "Inspect it with:" >&2
        echo "  nebius compute instance get --id \"$VM_ID\"" >&2
        echo >&2
        echo "Delete it, if required, with:" >&2
        echo "  nebius compute instance delete --id \"$VM_ID\"" >&2
    else
        echo "No VM ID had been returned when the error occurred." >&2
    fi

    echo "============================================================" >&2
    exit "$exit_code"
}

trap on_error ERR

for argument in "$@"; do
    case "$argument" in
        --yes)
            SKIP_CONFIRMATION=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $argument"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

# Ensure Nebius CLI is on PATH (installed to ~/.nebius/bin)
if [ -f "$HOME/.nebius/path.bash.inc" ]; then
    source "$HOME/.nebius/path.bash.inc"
fi

command -v nebius >/dev/null 2>&1 || fail \
"Nebius CLI is not installed.

Install and configure it with:
  curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  nebius profile create"

command -v jq >/dev/null 2>&1 || fail \
"jq is not installed.

On Ubuntu or WSL, install it with:
  sudo apt-get update
  sudo apt-get install -y jq"

command -v ssh-keygen >/dev/null 2>&1 || fail "ssh-keygen is not installed."

if [[ ! -f "$PRIVATE_KEY" ]]; then
    fail "Private SSH key not found: $PRIVATE_KEY"
fi

if [[ ! -f "$PUBLIC_KEY" ]]; then
    echo "Public key not found. Recreating it from the private key..."
    ssh-keygen -y -f "$PRIVATE_KEY" > "$PUBLIC_KEY"
    chmod 644 "$PUBLIC_KEY"
fi

ssh-keygen -lf "$PUBLIC_KEY" >/dev/null 2>&1 || \
    fail "The public SSH key is invalid: $PUBLIC_KEY"

chmod 600 "$PRIVATE_KEY"

if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="$(nebius config get parent-id 2>/dev/null | tr -d '\r\n')"
fi

# If no project ID stored, or the stored one fails, auto-detect from account
if [[ -z "$PROJECT_ID" ]] || ! nebius iam project get --id "$PROJECT_ID" --format json >/dev/null 2>&1; then
    echo "Stored project ID is missing or invalid. Auto-detecting..."
    # Use tenant ID to list projects (parent-id may be stale/empty)
    nebius config set parent-id "$TENANT_ID" 2>/dev/null || true
    DETECTED_ID="$(nebius iam project list --format json 2>/dev/null \
        | jq -r '.items[] | select(.metadata.name == "'"$EXPECTED_PROJECT_NAME"'") | .metadata.id // empty' \
        | head -1)"

    if [[ -n "$DETECTED_ID" ]]; then
        echo "Found project '$EXPECTED_PROJECT_NAME' with ID: $DETECTED_ID"
        nebius config set parent-id "$DETECTED_ID"
        PROJECT_ID="$DETECTED_ID"
    else
        fail "No Nebius project ID is configured and auto-detect found no project named '$EXPECTED_PROJECT_NAME'.

Set it manually with:
  nebius config set parent-id <PROJECT_ID>

Or override: PROJECT_ID=<id> ./nebius_provision.sh"
    fi
fi

# ---------------------------------------------------------------------------
# Validate the active project
# ---------------------------------------------------------------------------

echo "Checking the active Nebius project..."

PROJECT_JSON="$(
    nebius iam project get \
        --id "$PROJECT_ID" \
        --format json
)"

ACTUAL_PROJECT_NAME="$(jq -r '.metadata.name // empty' <<<"$PROJECT_JSON")"

if [[ -z "$ACTUAL_PROJECT_NAME" ]]; then
    fail "Nebius returned the project, but its name could not be read."
fi

if [[ "$ACTUAL_PROJECT_NAME" != "$EXPECTED_PROJECT_NAME" ]]; then
    fail "The active Nebius project is not the expected project.

Expected project: $EXPECTED_PROJECT_NAME
Actual project:   $ACTUAL_PROJECT_NAME
Project ID:       $PROJECT_ID

Change the active project or override EXPECTED_PROJECT_NAME deliberately."
fi

# ---------------------------------------------------------------------------
# Resolve the service account and subnet
# ---------------------------------------------------------------------------

echo "Resolving service account: $SERVICE_ACCOUNT_NAME"

SERVICE_ACCOUNT_JSON="$(
    nebius iam service-account get-by-name \
        --name "$SERVICE_ACCOUNT_NAME" \
        --parent-id "$PROJECT_ID" \
        --format json
)"

SERVICE_ACCOUNT_ID="$(jq -r '.metadata.id // empty' <<<"$SERVICE_ACCOUNT_JSON")"

if [[ -z "$SERVICE_ACCOUNT_ID" ]]; then
    fail "Service account not found: $SERVICE_ACCOUNT_NAME"
fi

if [[ -z "${SUBNET_ID:-}" ]]; then
    echo "Selecting the first subnet in project: $ACTUAL_PROJECT_NAME"

    SUBNET_JSON="$(
        nebius vpc subnet list \
            --parent-id "$PROJECT_ID" \
            --all \
            --format json
    )"

    SUBNET_ID="$(jq -r '.items[0].metadata.id // empty' <<<"$SUBNET_JSON")"
fi

if [[ -z "$SUBNET_ID" ]]; then
    fail "No subnet was found. Supply one explicitly, for example:

  SUBNET_ID=vpcsubnet-... ./nebius_provision.sh"
fi

# ---------------------------------------------------------------------------
# Prevent accidental duplicate GPU creation
# ---------------------------------------------------------------------------

EXISTING_VM_FILE="$(mktemp)"

if nebius compute instance get-by-name \
    --name "$VM_NAME" \
    --parent-id "$PROJECT_ID" \
    --format json \
    >"$EXISTING_VM_FILE" 2>/dev/null; then

    EXISTING_VM_ID="$(jq -r '.metadata.id // empty' "$EXISTING_VM_FILE")"

    EXISTING_IP="$(
        jq -r '
            .status.network_interfaces[0].public_ip_address.address
            // empty
            | split("/")[0]
        ' "$EXISTING_VM_FILE"
    )"

    rm -f "$EXISTING_VM_FILE"

    echo
    echo "A VM named '$VM_NAME' already exists."
    echo "VM ID: ${EXISTING_VM_ID:-unknown}"
    echo "Public IP: ${EXISTING_IP:-not currently available}"
    echo
    echo "The script will not create a second billable H100 VM."
    exit 1
fi

rm -f "$EXISTING_VM_FILE"

# ---------------------------------------------------------------------------
# Show the proposed expensive resource and ask for confirmation
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "NEBIUS H100 PROVISIONING PLAN"
echo "============================================================"
echo "Project:               $ACTUAL_PROJECT_NAME"
echo "Project ID:            $PROJECT_ID"
echo "VM name:               $VM_NAME"
echo "GPU platform:          $PLATFORM"
echo "Preset:                $PRESET"
echo "Boot disk:             ${BOOT_DISK_SIZE_GIB} GiB managed SSD"
echo "Boot image:            $IMAGE_FAMILY"
echo "Static public IP:      Auto assign"
echo "SSH username:          $VM_USER"
echo "Private key on laptop: $PRIVATE_KEY"
echo "Public key uploaded:   $PUBLIC_KEY"
echo "Service account:       $SERVICE_ACCOUNT_NAME"
echo "Service account ID:    $SERVICE_ACCOUNT_ID"
echo "Subnet ID:             $SUBNET_ID"
echo "============================================================"

if [[ "$SKIP_CONFIRMATION" != true ]]; then
    echo
    read -r -p "Type CREATE to provision this billable H100 VM: " confirmation

    if [[ "$confirmation" != "CREATE" ]]; then
        echo "Cancelled. No VM was created."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Build cloud-init and networking specifications
# ---------------------------------------------------------------------------

USER_DATA="$(
    jq -Rrs '.' <<CLOUD_INIT
#cloud-config
users:
  - name: $VM_USER
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$PUBLIC_KEY")
CLOUD_INIT
)"

NETWORK_INTERFACES="$(
    jq -cn \
        --arg subnet_id "$SUBNET_ID" \
        '[{
            "name": "eth0",
            "subnet_id": $subnet_id,
            "ip_address": {},
            "public_ip_address": {
                "static": true
            }
        }]'
)"

# ---------------------------------------------------------------------------
# Create and start the VM
# ---------------------------------------------------------------------------

echo
echo "Creating and starting the H100 VM..."
echo "This can take several minutes if capacity is available."

VM_JSON="$(
    nebius compute instance create \
        --parent-id "$PROJECT_ID" \
        --name "$VM_NAME" \
        --hostname "$VM_NAME" \
        --resources-platform "$PLATFORM" \
        --resources-preset "$PRESET" \
        --service-account-id "$SERVICE_ACCOUNT_ID" \
        --boot-disk-attach-mode read_write \
        --boot-disk-managed-disk-name "$BOOT_DISK_NAME" \
        --boot-disk-managed-disk-size-gibibytes "$BOOT_DISK_SIZE_GIB" \
        --boot-disk-managed-disk-type "$BOOT_DISK_TYPE" \
        --boot-disk-managed-disk-block-size-bytes 4096 \
        --boot-disk-managed-disk-source-image-family-image-family "$IMAGE_FAMILY" \
        --cloud-init-user-data "$USER_DATA" \
        --network-interfaces "$NETWORK_INTERFACES" \
        --timeout 20m \
        --format json
)"

VM_ID="$(jq -r '.metadata.id // empty' <<<"$VM_JSON")"

if [[ -z "$VM_ID" ]]; then
    fail "Nebius did not return a VM ID."
fi

echo "VM created."
echo "VM ID: $VM_ID"
echo "Waiting for the static public IP address..."

# ---------------------------------------------------------------------------
# Wait for the public IP address
# ---------------------------------------------------------------------------

for _ in $(seq 1 60); do
    INSTANCE_JSON="$(
        nebius compute instance get \
            --id "$VM_ID" \
            --format json
    )"

    PUBLIC_IP="$(
        jq -r '
            .status.network_interfaces[0].public_ip_address.address
            // empty
            | split("/")[0]
        ' <<<"$INSTANCE_JSON"
    )"

    if [[ -n "$PUBLIC_IP" ]]; then
        break
    fi

    sleep 5
done

if [[ -z "$PUBLIC_IP" ]]; then
    fail "The VM exists, but no public IP address appeared within five minutes."
fi

# ---------------------------------------------------------------------------
# Save the VM details locally
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$STATE_FILE")"

{
    printf 'VM_ID=%q\n' "$VM_ID"
    printf 'VM_NAME=%q\n' "$VM_NAME"
    printf 'PROJECT_ID=%q\n' "$PROJECT_ID"
    printf 'PROJECT_NAME=%q\n' "$ACTUAL_PROJECT_NAME"
    printf 'PUBLIC_IP=%q\n' "$PUBLIC_IP"
    printf 'SSH_HOST_ALIAS=%q\n' "$VM_HOST_ALIAS"
    printf 'SSH_USER=%q\n' "$VM_USER"
    printf 'PRIVATE_KEY=%q\n' "$PRIVATE_KEY"
    printf 'PUBLIC_KEY=%q\n' "$PUBLIC_KEY"
    printf 'SERVICE_ACCOUNT_NAME=%q\n' "$SERVICE_ACCOUNT_NAME"
    printf 'SERVICE_ACCOUNT_ID=%q\n' "$SERVICE_ACCOUNT_ID"
    printf 'SUBNET_ID=%q\n' "$SUBNET_ID"
    printf 'BOOT_DISK_NAME=%q\n' "$BOOT_DISK_NAME"
    printf 'BOOT_DISK_SIZE_GIB=%q\n' "$BOOT_DISK_SIZE_GIB"
} > "$STATE_FILE"

chmod 600 "$STATE_FILE"

# ---------------------------------------------------------------------------
# Replace the existing "Host nebius-vm" block in ~/.ssh/config
# ---------------------------------------------------------------------------

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$SSH_CONFIG"

SSH_BACKUP="${SSH_CONFIG}.bak.$(date +%Y%m%d_%H%M%S)"
cp "$SSH_CONFIG" "$SSH_BACKUP"

TEMP_CONFIG="$(mktemp)"

awk -v target="$VM_HOST_ALIAS" '
    BEGIN {
        skipping = 0
    }

    /^[[:space:]]*Host[[:space:]]+/ {
        if (skipping) {
            skipping = 0
        }

        if (NF == 2 && $1 == "Host" && $2 == target) {
            skipping = 1
            next
        }
    }

    !skipping {
        print
    }
' "$SSH_CONFIG" > "$TEMP_CONFIG"

cat >> "$TEMP_CONFIG" <<SSH_BLOCK

Host $VM_HOST_ALIAS
    HostName $PUBLIC_IP
    IdentityFile $PRIVATE_KEY
    IdentitiesOnly yes
    User $VM_USER
    Port 22
    Compression yes
    ServerAliveInterval 60
    ServerAliveCountMax 10
    StrictHostKeyChecking accept-new
SSH_BLOCK

mv "$TEMP_CONFIG" "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# A recycled cloud public IP can already have an obsolete host key locally.
ssh-keygen -R "$PUBLIC_IP" >/dev/null 2>&1 || true
ssh-keygen -R "$VM_HOST_ALIAS" >/dev/null 2>&1 || true

KEY_FINGERPRINT="$(ssh-keygen -lf "$PUBLIC_KEY" | awk '{print $2}')"

# ---------------------------------------------------------------------------
# Clear acknowledgement: the VM and static IP now exist
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "NEBIUS H100 VM CREATED AND STARTED"
echo "============================================================"
echo "Project name:          $ACTUAL_PROJECT_NAME"
echo "Project ID:            $PROJECT_ID"
echo
echo "VM name:               $VM_NAME"
echo "VM ID:                 $VM_ID"
echo "GPU platform:          $PLATFORM"
echo "GPU preset:            $PRESET"
echo "Boot disk:             ${BOOT_DISK_SIZE_GIB} GiB managed SSD"
echo
echo "STATIC PUBLIC IP:      $PUBLIC_IP"
echo
echo "SSH host alias:        $VM_HOST_ALIAS"
echo "SSH username:          $VM_USER"
echo "Private key on laptop: $PRIVATE_KEY"
echo "Public key supplied:   $PUBLIC_KEY"
echo "Key fingerprint:       $KEY_FINGERPRINT"
echo
echo "Service account:       $SERVICE_ACCOUNT_NAME"
echo "Service account ID:    $SERVICE_ACCOUNT_ID"
echo
echo "SSH config updated:    $SSH_CONFIG"
echo "SSH config backup:     $SSH_BACKUP"
echo "State file:            $STATE_FILE"
echo
echo "Connect using:"
echo "  ssh $VM_HOST_ALIAS"
echo
echo "Direct equivalent:"
echo "  ssh -i \"$PRIVATE_KEY\" \"$VM_USER@$PUBLIC_IP\""
echo "============================================================"

# ---------------------------------------------------------------------------
# Wait for the SSH port, without requiring an unencrypted private key
# ---------------------------------------------------------------------------

echo
echo "Waiting for TCP port 22 to become reachable..."

SSH_PORT_READY=false

for _ in $(seq 1 60); do
    if timeout 5 bash -c "true </dev/tcp/$PUBLIC_IP/22" 2>/dev/null; then
        SSH_PORT_READY=true
        break
    fi

    sleep 5
done

if [[ "$SSH_PORT_READY" == true ]]; then
    echo
    echo "SSH port 22 is reachable."
    echo
    echo "Next command:"
    echo "  ssh $VM_HOST_ALIAS"
else
    echo
    echo "WARNING: The VM and static IP were created successfully, but port 22"
    echo "did not become reachable within five minutes."
    echo
    echo "Do not run this provisioning script again."
    echo "Inspect the existing VM with:"
    echo "  nebius compute instance get --id \"$VM_ID\""
fi

echo
echo "When the GPU work is finished, first copy your results back."
echo "Then delete the VM synchronously with:"
echo
echo "  source \"$STATE_FILE\""
echo "  nebius compute instance delete --id \"\$VM_ID\""
echo
echo "The managed 200 GiB boot disk and VM-lifetime static IP should be"
echo "deleted with the VM."
