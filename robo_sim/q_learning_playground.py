import mr_robot
import env_manager

robots = mr_robot.Simple_q_learner(1, 20, [400, 700], 1, 1)
environment = env_manager.env_manager("env_config/env2.json", [robots])
environment.start_simulation()