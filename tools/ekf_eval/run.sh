#!/bin/bash
# Run ekf_localization_ros2 in Gazebo with noisy odometry, drive the loop, log the CSV.
#   run.sh [max_seconds]
T=$(cd "$(dirname "$0")" && pwd)
OUT=$T/out; mkdir -p $OUT
MAX=${1:-240}
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source $T/../../install/setup.bash
export TURTLEBOT3_MODEL=burger

setsid ros2 launch ekf_localization ekf_localization.launch.py odom_topic:=/odom_noisy > $OUT/launch.log 2>&1 &
LP=$!
python3 $T/ekf_tools.py noisy_odom > $OUT/noisy.log 2>&1 &
NP=$!
# EKF starts 7 s after launch and needs the landmark list; give it time
for i in $(seq 60); do grep -qa "ekf_localization_node" $OUT/launch.log && break; sleep 1; done
sleep 15
echo "stack up, driving"
python3 $T/ekf_tools.py log $OUT/ekf-paths.csv $MAX > $OUT/logger.log 2>&1 &
LG=$!
sleep 2
timeout $MAX python3 $T/ekf_tools.py drive
sleep 3
kill -INT $LG; wait $LG; cat $OUT/logger.log

# Are the cylinder markers actually being published?
for tp in /reference_cylinder_markers /estimated_cylinder_markers; do
    n=$(timeout 6 ros2 topic echo --once $tp 2>/dev/null | grep -c "ns:")
    echo "$tp: ${n:-0} markers"
done
grep -qa "AttributeError\|Traceback" <(grep -a "localization-" $OUT/launch.log) && echo "EKF NODE CRASHED (see launch.log)"

# RViz window grab (covariance ellipse etc.)
WID=$(xwininfo -root -tree | grep -i '"rviz' | head -1 | awk '{print $1}')
[ -n "$WID" ] && xwd -id $WID -out $OUT/rviz.xwd && ffmpeg -y -v error -i $OUT/rviz.xwd $OUT/rviz.png && rm $OUT/rviz.xwd

kill $NP; kill -INT -$LP 2>/dev/null; sleep 4; kill -9 -$LP 2>/dev/null
pkill -f gzserver; pkill -f gzclient; pkill -f rviz2
grep -aiE "error|exception|traceback" $OUT/launch.log | grep -viE "objectpose|warn" | head -5
