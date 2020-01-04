import mr_robot
import env_manager

robots = mr_robot.Manual_control_robot(1, 20, [200, 200], 1, 1)
environment = env_manager.env_manager("env_config/env1.json", [robots])
environment.start_simulation()