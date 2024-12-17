# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:20:48 2024

@author: camde
"""

import math

r = 0.9995
theta1 = math.pi
theta2 = math.pi - math.asin((r) / 1 * math.sin(theta1))
d = math.sqrt((r) ** 2 + 1 ** 2 - 2 * (r) * 1 * math.cos(theta2 - theta1))
x = (r) ** 2 + 1 ** 2 - 2 * (r) * 1 * math.cos(theta2 - theta1)

print(d)
print(math.sqrt(0.9995 ** 2 + 1 - 2 * 0.9995))