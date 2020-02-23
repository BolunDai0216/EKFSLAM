import numpy as np
from pdb import set_trace


class Landmark():
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class OdometryData():
    def __init__(self, r1, t, r2):
        self.r1 = float(r1)
        self.t = float(t)
        self.r2 = float(r2)


class SensorData():
    def __init__(self, id, range, bearing):
        self.id = int(id)
        self.range = float(range)
        self.bearing = float(bearing)


class Data():
    def __init__(self):
        self.odometry_data = None
        self.sensor_data = []

    def get_odometry_data(self, data):
        self.odometry_data = OdometryData(data[1], data[2], data[3])

    def get_sensor_data(self, data):
        self.sensor_data.append(SensorData(data[1], data[2], data[3]))


def read_world(filename="data/world.dat"):
    data = np.loadtxt(filename)
    landmarks = []
    for d in data:
        landmarks.append(Landmark(int(d[0]), d[1], d[2]))
    return landmarks


def read_data(filename="data/sensor_data.dat"):
    timesteps = []
    with open(filename) as fn:
        data = fn.readlines()
    for d in data:
        txt = d.split()
        if txt[0] == "ODOMETRY":
            timesteps.append(Data())
            timesteps[-1].get_odometry_data(txt)
        else:
            timesteps[-1].get_sensor_data(txt)
    return timesteps


def main():
    timesteps = read_data()
    print(timesteps[0].odometry_data.r1)
    set_trace()


if __name__ == "__main__":
    main()
