from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    cmd = (
        "bash -c '"
        "source /opt/ros/humble/setup.bash && "
        "source /home/sidibe/vision_ws/install/setup.bash && "
        "source /home/sidibe/yolo_project/yolo_env/bin/activate && "
        "python /home/sidibe/vision_ws/src/yolo_detector/yolo_detector/pvclip_fall_node.py"
        "'"
    )
    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '-c', cmd],
            output='screen'
        )
    ])