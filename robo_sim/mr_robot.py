import numpy as np
from shapely import geometry, affinity
from abc import ABC, abstractmethod 

class Mr_robot(ABC):
	def __init__(self, id_num, size, starting_loc, health, max_health):
		self.id = id_num
		self.size = size
		self.angle = 0
		self.location = np.array(starting_loc).astype(np.float64)
		self.starting_loc = np.array(starting_loc).astype(np.float64)
		self.polygon = self.build_polygon()
		self.sensor_contacts = []
		self.food_sensed = [] # 1 if sensor is pointed at food
		self.health = health
		self.max_health = max_health
		self.food_flag = False
		self.life_count = 0
		self.high_score = 0
		self.current_score = 0
		self.contact_score_modifier = 0

	def build_polygon(self):
		offset = self.size//2
		points = [self.location + np.array([-offset, -offset]),
				  self.location + np.array([-offset, offset]),
				  self.location + np.array([offset, offset]),
				  self.location + np.array([offset, -offset])]
		return geometry.Polygon(points)

	def apply_rotation(self, rotation):
		self.angle += rotation
		self.polygon = affinity.rotate(self.polygon, rotation)

	def travel(self, dist, env):
		y_offset = -np.cos(np.radians(self.angle)) * dist
		x_offset = np.sin(np.radians(self.angle)) * dist

		new_position = self.location + np.array([x_offset, y_offset])
		travel_candidate_line = geometry.LineString([self.location, new_position])
		for wall in env["shapely_walls"]:
			collision = wall.intersection(travel_candidate_line)
			if collision: 
				# no update performed
				self.contact_score_modifier = -2
				return

		self.polygon = affinity.translate(self.polygon, x_offset, y_offset)
		self.location += np.array([x_offset, y_offset])
		self.contact_score_modifier = 0

	def get_sensor_inputs(self, env, food):
		self.sensor_contacts = []
		self.food_sensed = []

		sensor_range = env["sensor_range"]
		for sensor_angle in env["sensor_angles"]:
			# calculate offsets for sensor coords
			absolute_angle = self.angle + sensor_angle
			y_offset = -np.cos(np.radians(absolute_angle)) * sensor_range
			x_offset = np.sin(np.radians(absolute_angle)) * sensor_range

			sensor_max = self.location + np.array([x_offset, y_offset]) # point of max dist of sensor
			max_sensor_ray = geometry.LineString([self.location, sensor_max]) # shapely line obect for sensor
			min_collision_point = {"point": sensor_max, "dist": sensor_range} # keep track of min distance - we want closest contact point
			
			# find min distance to a wall
			for wall in env["shapely_walls"]:
				collision = wall.intersection(max_sensor_ray)
				if collision:
					contact_point = [collision.xy[0][0], collision.xy[1][0]]
					dist = np.linalg.norm(np.abs(np.array(contact_point) - self.location))
					if dist < min_collision_point["dist"]:
						min_collision_point["point"] = contact_point
						min_collision_point["dist"] = dist
			self.sensor_contacts.append(min_collision_point["point"])
			self.food_sensed.append(0)

			# find min distance to a food obj
			for food_entity in food:
				collision = food_entity.intersection(max_sensor_ray)
				if collision:
					contact_point = [collision.xy[0][0], collision.xy[1][0]]
					dist = np.linalg.norm(np.abs(np.array(contact_point) - self.location))
					if dist < min_collision_point["dist"]:
						min_collision_point["point"] = contact_point
						min_collision_point["dist"] = dist
						self.food_sensed[-1] = 1
						self.sensor_contacts[-1] = min_collision_point["point"]

	def get_food_collision(self, food):
		for food_poly in food:
			if food_poly.intersection(self.polygon):
				self.food_flag = True
				print("\t\t\t\tgot food!!!!")
				return food_poly
		return []
		
	def update(self, env, food):
		ret = {}

		# update health
		self.health -= 1
		print(f"Bot {self.id} health: {self.health}")
		print(f"epsilon: {round(self.epsilon,4)}   Life: {self.life_count}")
		print(f"High score: {self.high_score}   current score: {self.current_score}")

		self.get_sensor_inputs(env, food)
		food_collected = self.get_food_collision(food)
		if food_collected:
			ret["food"] = food_collected
			self.health += env["food_value"]
			if self.health > env["max_health"]:
				self.health = env["max_health"]
		return ret

	@abstractmethod
	def act(self):
		pass

class Manual_control_robot(Mr_robot):
	def __init__(self, id_num, size, starting_loc, health, max_health):
		super().__init__(id_num, size, starting_loc, health, max_health)

	def control_input(self, key, env):
		if key == 97: # a 
			self.apply_rotation(-10)
		if key == 100: # d
			self.apply_rotation(10)
		if key == 119: # w
			self.travel(20, env)
		if key == 115: # s
			self.travel(-20, env)

class Simple_q_learner(Mr_robot):
	def __init__(self, id_num, size, starting_loc, health, max_health):
		super().__init__(id_num, size, starting_loc, health, max_health)

	def init_env_data(self, env):
		print("in env init.")
		# state encoding: food visible for each sensor, past 4 moves, actions
		self.learning_rate = env["q_settings"]["lr"]
		self.discount = env["q_settings"]["discount"]
		self.action_history_length = env["q_settings"]["action_history"]
		self.sensor_count = len(env["sensor_angles"])
		self.epsilon = env["q_settings"]["epsilon"]
		self.epsilon_decay = env["q_settings"]["epsilon_decay"]
		self.range_buckets = env["q_settings"]["range_buckets"]
		self.sensor_range = env["sensor_range"]

		q_shape = ([2] * self.sensor_count + [4]*self.action_history_length + [self.range_buckets, 4])
		self.q_table = np.zeros(q_shape)
		#self.q_table = np.random.random(q_shape)
		self.action_history = [0]*self.action_history_length # [up, down, left, right]
		self.last_state_action = []

	def get_discrete_state(self):
		middle_sensor_idx = self.sensor_count//2
		dist = np.linalg.norm(np.abs(self.sensor_contacts[middle_sensor_idx] - self.location))
		range_state = int(dist/self.sensor_range * self.range_buckets)
		state = self.food_sensed + self.action_history + [range_state -1]
		return tuple(np.array(state).astype(np.int))

	def act(self, env):
		state = self.get_discrete_state()

		if np.random.random() > self.epsilon: # select action from q table
			action = np.argmax(self.q_table[state])
		else: # select random action
			action = int(np.random.random()*4)

		# perform action
		if action == 0: # left
			self.apply_rotation(-5)
		if action == 1: # right
			self.apply_rotation(5)
		if action == 2: # up
			self.travel(20, env)
		if action == 3: # down
			self.travel(-20, env)
		print("ACTION: ", action)

		# update based on last action
		if self.last_state_action: # if this isn't the first move
			if self.food_flag:
				self.q_table[self.last_state_action] += 1000
				self.food_flag = False
			else:
				self.q_table[self.last_state_action] -= 1
			self.q_table[self.last_state_action] += self.contact_score_modifier
			best_action_index = np.argmax(self.q_table[state])
			best_action_score = self.q_table[state][best_action_index]
			self.q_table[self.last_state_action] += best_action_index * self.discount

		self.action_history = self.action_history[1:]
		self.action_history.append(action)

		# calculate last state and action
		middle_sensor_idx = self.sensor_count//2
		dist = np.linalg.norm(np.abs(self.sensor_contacts[middle_sensor_idx] - self.location))
		range_state = int(dist/self.sensor_range * self.range_buckets)

		self.last_state_action = tuple(np.array(self.food_sensed + \
												self.action_history + \
												[range_state-1] + \
												[action]).astype(np.int))
		self.current_score += 1
		if self.high_score < self.current_score: self.high_score = self.current_score


	def kill(self, env):
		# self.epsilon -= self.epsilon_decay
		# self.action_history = [0]*self.action_history_length
		# self.location = self.starting_loc
		# self.angle = 0
		# self.build_polygon()
		# self.health = env["starting_health"]
		self.health = env["starting_health"]
		self.epsilon -= self.epsilon_decay
		if self.epsilon <= 0: self.epsilon = 0
		self.life_count += 1
		self.current_score = 0
		self.q_table[self.last_state_action] -= 20








if __name__ == "__main__":
	x = Manual_control_robot(1, 20, [200,200])
	print(x.id)
	print(x.polygon)
	# x.update([])
	# print(x.__class__.__name__)
