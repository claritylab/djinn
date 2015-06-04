cpus=$1

# Enable number of cpus as desired
if [ $cpus = 1 ]; then

  echo "Turn on 1 core"

  echo 1 > /sys/devices/system/cpu/cpu0/online
  echo 0 > /sys/devices/system/cpu/cpu1/online
  echo 0 > /sys/devices/system/cpu/cpu2/online
  echo 0 > /sys/devices/system/cpu/cpu3/online

elif [ $cpus = 2 ]; then

  echo "Turn on 2 cores"

  echo 1 > /sys/devices/system/cpu/cpu0/online
  echo 1 > /sys/devices/system/cpu/cpu1/online
  echo 0 > /sys/devices/system/cpu/cpu2/online
  echo 0 > /sys/devices/system/cpu/cpu3/online

elif [ $cpus = 3 ]; then

  echo "Turn on 3 cores"
  
  echo 1 > /sys/devices/system/cpu/cpu0/online
  echo 1 > /sys/devices/system/cpu/cpu1/online
  echo 1 > /sys/devices/system/cpu/cpu2/online
  echo 0 > /sys/devices/system/cpu/cpu3/online
    
elif [ $cpus = 4 ]; then

  echo "Turn on 4 cores"
  
  echo 1 > /sys/devices/system/cpu/cpu0/online
  echo 1 > /sys/devices/system/cpu/cpu1/online
  echo 1 > /sys/devices/system/cpu/cpu2/online
  echo 1 > /sys/devices/system/cpu/cpu3/online

fi

sleep 5
