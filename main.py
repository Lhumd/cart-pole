import sys
import numpy as np
import controller
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


class Pos:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Pole:
    def __init__(self, pos):
        self.length = 1
        self.angle = 1.0
        self.firstPos = Pos(pos.x, pos.y - 50)
        self.secondPos = Pos()
        self.segment(pos)
        self.velocity = 0
        self.pastVelocity = 0
        self.previousAngle = self.angle
        self.acceleration = 0
        self.mass = 1

    def segment(self, pos):
        self.firstPos.x = pos.x
        self.firstPos.y = pos.y - 50
        self.secondPos.x = self.firstPos.x - np.sin(self.angle) * self.length * 150
        self.secondPos.y = self.firstPos.y - np.cos(self.angle) * self.length * 150


class Cart:
    def __init__(self):
        self.pos = Pos(500.0, 900.0)
        self.previousPos = self.pos
        self.pole = Pole(self.pos)
        self.acceleration = 0
        self.velocity = 0
        self.pastVelocity = 0
        self.mass = 5



def display(cart):
    img = np.zeros((1000, 1000, 3), np.uint8)
    cv2.line(img, (int(cart.pole.firstPos.x), int(cart.pole.firstPos.y)), (int(cart.pole.secondPos.x), int(cart.pole.secondPos.y)), (0, 255, 0), 3)
    cv2.rectangle(img, (int(cart.pos.x - 80), int(cart.pos.y - 50)), (int(cart.pos.x + 80), int(cart.pos.y + 50)), (0, 255, 0), 15)
    cv2.imshow('display', img)
    cv2.waitKey(5)


def add_force(cart, f, t):
    g = 9.81
    cart.pole.pastVelocity = cart.pole.velocity
    cart.pole.velocity = (cart.pole.angle - cart.pole.previousAngle) / t
    cart.pastVelocity = cart.velocity
    cart.velocity = (cart.pos.x - cart.previousPos.x) / t
    cart.pole.acceleration = (f + (g * np.sin(cart.pole.angle) * (cart.pole.mass + cart.mass) / np.cos(cart.pole.angle))
                              - (cart.pole.mass * cart.pole.length * (cart.pole.velocity ** 2) * np.sin(cart.pole.angle))) /\
                             ((cart.pole.length * (cart.pole.mass + cart.mass) / np.cos(cart.pole.angle)) -
                              (cart.pole.mass * cart.pole.length * np.cos(cart.pole.angle)))
    cart.acceleration = (cart.pole.length * cart.pole.acceleration - g * np.sin(cart.pole.angle)) / np.cos(cart.pole.angle)
    a = cart.pos.x
    cart.pos = Pos(cart.pos.x + ((t ** 2) * cart.acceleration) + (cart.pos.x - cart.previousPos.x), cart.pos.y)
    cart.previousPos.x = a
    cart.pole.previousAngle = cart.pole.angle
    cart.pole.angle += ((t ** 2) * cart.pole.acceleration) + (t * cart.pole.velocity)
    cart.pole.segment(cart.pos)


def find_force(cart):
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 9.81 * cart.pole.mass / cart.mass, 0],
        [0, 0, 0, 1],
        [0, 0, (cart.mass + cart.pole.mass) * 9.81 / (cart.pole.length * cart.mass), 0]
    ])
    B = np.matrix([[0], [1 / cart.mass], [0], [1 / (cart.pole.length * cart.mass)]])
    Q = np.matrix([
        [10, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 10000, 0],
        [0, 0, 0, 1]
    ])
    R = np.matrix([500])
    K = controller.LQR(A, B, Q, R)
    np.matrix(K)
    x = np.matrix([
        [np.squeeze(np.asarray(cart.pos.x))],
        [np.squeeze(np.asarray(cart.velocity))],
        [np.squeeze(np.asarray(cart.pole.angle))],
        [np.squeeze(np.asarray(cart.pole.velocity))]
    ])
    desired = np.matrix([[500], [0], [0], [0]])
    F = -(K * (x - desired))
    print(np.squeeze(np.asarray(F)))
    return np.squeeze(np.asarray(F))


if __name__ == "__main__":
    cart = Cart()
    t = 0.01
    while True:
        display(cart)
        add_force(cart, find_force(cart), t)
