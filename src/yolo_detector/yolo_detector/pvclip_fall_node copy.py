import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class Publisher(Node):
    def __init__(self):
        super().__init__("publisher")

        self.pub = self.create_publisher(Twist , "cmd_vel" , 10)
        self.odom = self.subscriptions(Odometry , "/odom"  , self.callodom)
        self.timer = self.create_timer(0.1,self.callback)


    def callback(self):
        msg = Twist()
        msg.linear.x = 2.5
        msg.angular.z = 0
        self.pub.publish(msg)


    def callodom(self , msg ):
        pose = Odometry()
        pos = pose.pose.position
        orientation  = pose.pose.orientation

        self.get_logger().info(f"x : {pos.x:.2f}  y : {pos.y:.2f}  z :{pos.z:.2f} ")

        if pos.x >=5.0:
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            



def main():
    rclpy.init()
    publier = Publisher()
    rclpy.spin(publier)
    publier.destroy_node()
    rclpy.shutdown()

