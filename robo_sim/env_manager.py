import cv2
import numpy as np
import json
from ast import literal_eval
from copy import copy
from shapely import geometry
import random

def to_int_tuple(data):
	return tuple([int(x) for x in data])

class env_manager:
	def __init__(self, env_file_string, robots):
		self.env = self.load_env(env_file_string)
		self.img_background = self.create_background_img()
		self.robots = robots
		self.food_count = 0
		self.food = []
		self.food = self.replenish_food()
		self.iterations = 0

	def replenish_food(self):
		new_food = []
		new_food.extend(self.food)

		x_min = self.env["boarder_padding"] // 2
		x_max = x_min + self.env["dimensions"][0]
		y_min = x_min
		y_max = y_min + self.env["dimensions"][1]
		offset = self.env["food_size"] // 2

		while self.food_count < self.env["food_count"]:
			# generage location for food
			x_loc = int(random.randrange(x_min, x_max))
			y_loc = int(random.randrange(y_min, y_max))
			center = [x_loc, y_loc]
			points = [center + np.array([-offset, -offset]),
					  center + np.array([-offset, offset]),
					  center + np.array([offset, offset]),
					  center + np.array([offset, -offset])]
			food_candidate = geometry.Polygon(points)

			for wall in self.env["shapely_walls"]:
				collision = wall.intersection(food_candidate)
				# if there is any collision try again
				if collision: continue

			new_food.append(food_candidate)
			self.food_count += 1
		return new_food

	def load_env(self, env_file_string):
		data = json.load(open(env_file_string))
		padding = data["boarder_padding"]//2
		width, height = data['dimensions']

		#compensate for the boarder padding and edit existing walls
		for i in range(len(data["walls"])):
			data["walls"][i][0][0] += padding
			data["walls"][i][0][1] += padding
			data["walls"][i][1][0] += padding
			data["walls"][i][1][1] += padding

		#add walls on bounds of image
		boundaries = [[[padding, padding], [width + padding, padding]],
					  [[padding, padding], [padding, height + padding]],
					  [[width + padding, height + padding], [width + padding, padding]],
					  [[width + padding, height + padding], [padding, height + padding]]]
		data["walls"].extend(boundaries)
		data["shapely_walls"] = [geometry.LineString(wall) for wall in data["walls"]]

		return data

	def create_background_img(self):
		img_dims = self.env["dimensions"]
		img_dims = [x + self.env["boarder_padding"] for x in img_dims] # apply expansion to img size 
		img = np.ones((img_dims[1], img_dims[0], 3), np.uint8)*255 # create white image

		# draw walls
		padding = np.array([self.env["boarder_padding"], self.env["boarder_padding"]])/2
		for wall in self.env["walls"]:
			wall = [[int(x) for x in point] for point in wall]
			img = cv2.line(img,
						   tuple(wall[0]),
						   tuple(wall[1]),
						   color=(255, 50, 25),
						   thickness=3)
		return img

	def render(self):
		img = copy(self.img_background)

		# draw robots ##############################
		padding = self.env["boarder_padding"]
		for robot in self.robots:
			# draw sensor contacts
			for contact in robot.sensor_contacts:
				img = cv2.line(img,
							   to_int_tuple(robot.location),
							   to_int_tuple(contact),
							   color=(10, 50, 255), thickness=2)

			# draw robots
			points = robot.polygon.exterior.coords.xy
			points = np.array([[[int(points[0][i]),
								 int(points[1][i])] for i in range(4)]])
			img = cv2.fillPoly(img, points, (100, 255, 0))

		# draw food #################################
		for food_poly in self.food:
			points = food_poly.exterior.coords.xy
			points = np.array([[[int(points[0][i]),
								 int(points[1][i])] for i in range(4)]])
			img = cv2.fillPoly(img, points, (200, 255, 10))

		# display image
		cv2.imshow("simulation window", img)

		## for manual control bots only, need to pass through controls
		if self.robots[0].__class__.__name__ == "Manual_control_robot":
			k = cv2.waitKey()
			if k == 27: exit()
			self.robots[0].control_input(k, self.env)

		else:
			k = cv2.waitKey(1)
			if k == 27: exit()

	def start_simulation(self):
		# initialize robots health
		for robot in self.robots:
			robot.health = self.env["starting_health"]
			robot.max_health = self.env["max_health"]
			if robot.__class__.__name__ == "Simple_q_learner":
				robot.init_env_data(self.env)
			if robot.__class__.__name__ == "DQN_agent":
				robot.init_env_data(self.env)


		# update all bots
		for robot in self.robots:
			robot.update(self.env, self.food)

		while True:
			#if self.iterations % 1 == 1 and self.iterations > 1:
			self.render()

			# update all bots
			for robot in self.robots:
				robot.act(self.env)
				env_effect = robot.update(self.env, self.food)
				if 'food' in env_effect:
					self.food_count -= 1
					self.food.remove(env_effect["food"])
					self.food = self.replenish_food()

				if robot.health <= 0:
					robot.kill(self.env)
					print("ded")
					self.iterations += 1

				if robot.__class__.__name__ == "DQN_agent":
					robot.train()

if __name__ == "__main__":
	env = env_manager("env_config/env1.json",[])
	#env.render()