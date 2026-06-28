# # main.py
# class Point(object):
#     def __init__(self, x: int, y: int):
#         self.x = x
#         self.y = y

#     def __str__(self):
#         return f"x={self.x}, y={self.y}"

# p = Point(10,11)

# # print(p)

# class Student:
#     school = "ABC School"   # class attribute

#     def greet(self):         # method
#         print("Hello")

# print(Student.__dict__)


# main.py
from dataclasses import dataclass
from math import sqrt, pow

@dataclass
class Point:
    x: int = 0
    y: int = 0

    def distance_to(self, p) -> float:
        return p
        return sqrt(pow(p.x - self.x, 2) + pow(p.y - self.y, 2))

p = Point(2,2)
p1 = Point(4,4)

print(p.distance_to(p1))
print(distance_to(p1))