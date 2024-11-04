#!/bin/bash

# This script unlocks the full frequency range and sampling rate capabilities of the PlutoSDR.
# It is safe to use, with all commands documented and endorsed by Analog Devices, the manufacturer of the AD936x transceiver.

# Define SSH credentials
HOST="192.168.2.1"
USER="root"
PASSWORD="analog"

execute_ssh() {
  sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no $USER@$HOST "$1"
}

# Commands for firmware version 0.31 or lower
execute_ssh "fw_setenv attr_name compatible"
execute_ssh "fw_setenv attr_val ad9364"

# Commands for firmware version 0.32 and higher
execute_ssh "fw_setenv compatible ad9364"

echo "Rebooting Pluto...."
execute_ssh "reboot"
