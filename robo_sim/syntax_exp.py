class child():
	def __init__(self):
		self.x = "child"


class Parent():
	def __init__(self, num):
		self.x = "Parent"
		self.children = [child() for x in range(num)]
		for c in self.children:
			print(c.x)
			print(type(super(child)))

p = Parent(3)
