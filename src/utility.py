"""
This module contains general-use functions.
"""

import numpy as np

def seq_is_equal(a, b):
    """
    Helper function that checks if two sequences are equal (assuming they have
    the same length).
    """
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def angle_is_between(angle, a, b):
    """
    Takes in 3 angles (in radians) and returns 'a' <= 'angle' <= 'b'.
    """
    to_degrees = lambda rads: int(rads * 180 / np.pi)
    reduce_degrees = lambda degr: (360 + degr % 360) % 360
    angle = to_degrees(angle)
    a = to_degrees(a)
    b = to_degrees(b)
    angle = reduce_degrees(angle)
    a = reduce_degrees(a)
    b = reduce_degrees(b)
    if a < b:
        return a <= angle <= b
    return a <= angle or angle <= b

def find_angle(x1, y1, x2, y2):
    """
    Finds the angle between two points (in radians).
    """
    angle = (360 + int(np.arctan2(y1 - y2, x2 - x1) * 180 / np.pi) % 360) % 360
    return angle * np.pi / 180

def distance_between(x1, y1, x2, y2):
    """
    Calculates the distance between 2 points.
    """
    return np.hypot(x1 - x2, y1 - y2)
