
# This script is not designed to run as it is, 
# you need to execute every command seperately with sudo

freq=$1
echo $freq > /sys/kernel/debug/clock/override.gbus/rate

echo "Check frequency is correctly set"
cat /sys/kernel/debug/clock/gbus/rate
