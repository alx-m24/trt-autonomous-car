#include <memory>
#include <termios.h>
#include <unistd.h>
#include <cstdio>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

class KeyboardTeleop : public rclcpp::Node {
public:
    KeyboardTeleop() : Node("keyboard_teleop") {
        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        // Publish at a fixed rate so the car doesn't jerk between keypresses,
        // and so it stops automatically if input goes stale.
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20Hz
            std::bind(&KeyboardTeleop::publishLoop, this));

        last_key_time_ = this->now();

        setupTerminal();
        RCLCPP_INFO(this->get_logger(),
            "Keyboard teleop ready. WASD to drive, SPACE to stop, Q to quit.");
    }

    ~KeyboardTeleop() {
        restoreTerminal();
    }

private:
    void setupTerminal() {
        tcgetattr(STDIN_FILENO, &old_termios_);
        struct termios raw = old_termios_;
        raw.c_lflag &= ~(ICANON | ECHO);   // no line-buffering, no echo
        raw.c_cc[VMIN] = 0;                // non-blocking read
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }

    void restoreTerminal() {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
    }

    void publishLoop() {
        char c;
        bool got_key = (read(STDIN_FILENO, &c, 1) > 0);

        if (got_key) {
            last_key_time_ = this->now();
            switch (c) {
                case 'w': linear_ = std::min(linear_ + step_, max_linear_); break;
                case 's': linear_ = std::max(linear_ - step_, -max_linear_); break;
                case 'a': angular_ = std::min(angular_ + step_, max_angular_); break;
                case 'd': angular_ = std::max(angular_ - step_, -max_angular_); break;
                case ' ': linear_ = 0.0; angular_ = 0.0; break;
                case 'q':
                    RCLCPP_INFO(this->get_logger(), "Quit requested.");
                    rclcpp::shutdown();
                    return;
                default: break;
            }
        }

        // Safety: zero the command if no key pressed recently (terminal lost
        // focus, node hung, etc.) so the car doesn't drive off unattended.
        if ((this->now() - last_key_time_).seconds() > 0.3) {
            linear_ = 0.0;
            angular_ = 0.0;
        }

        geometry_msgs::msg::Twist msg;
        msg.linear.x = linear_;
        msg.angular.z = angular_;
        cmd_pub_->publish(msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    struct termios old_termios_;

    double linear_ = 0.0;
    double angular_ = 0.0;
    const double step_ = 0.05;
    const double max_linear_ = 1.0;
    const double max_angular_ = 1.5;
    rclcpp::Time last_key_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KeyboardTeleop>());
    rclcpp::shutdown();
    return 0;
}
