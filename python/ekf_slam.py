from utils import read_data, read_world
from utils import normalize_angle
import numpy as np
import math
from pdb import set_trace


class EKFSLAM():
    def __init__(self, landmarks, timesteps):
        self.landmarks = landmarks
        self.timesteps = timesteps
        self.N = len(landmarks)
        self.observed_landmarks = np.zeros(self.N, dtype=bool)
        self.mu = np.zeros((2*self.N+3, 1))
        self.sigma = np.identity(2*self.N+3) * 1e3
        self.sigma[0:3, 0:3] = np.zeros((3, 3))
        self.R = np.zeros_like(self.sigma)
        self.R[0:3, 0:3] = np.diag(np.array([0.1, 0.1, 0.01]))

    def prediction(self, step):
        r1 = self.timesteps[step].odometry_data.r1
        t = self.timesteps[step].odometry_data.t
        r2 = self.timesteps[step].odometry_data.r2
        prior_pos = self.mu[0:3]

        update = np.array([[t*math.cos(prior_pos[2]+r1)],
                           [t*math.sin(prior_pos[2]+r1)],
                           [r1+r2]])
        F = np.hstack((np.identity(3), np.zeros((3, 2*self.N))))
        self.mu = self.mu + np.matmul(np.transpose(F), update)
        self.mu[2] = normalize_angle(self.mu[2])

        Gx = np.array([[0, 0, -update[1]],
                       [0, 0, update[0]],
                       [0, 0, 0]])
        G = np.identity(self.N*2+3) + np.matmul(np.transpose(F), np.matmul(Gx, F))

        self.sigma = np.matmul(G, np.matmul(self.sigma, np.transpose(G))) + self.R

    def correction(self, step):
        sensor_data = self.timesteps[step].sensor_data
        m = len(sensor_data)
        Z = np.zeros((2*m, 1))
        expectedZ = np.zeros((2*m, 1))
        H = np.zeros((2*m, 2*self.N+3))

        for data in sensor_data:
            landmark_id = data.id
            if self.observed_landmarks[landmark_id] is False:
                self.mu[2*landmark_id+1] = self.mu[0] + \
                    data.range * math.cos(self.mu[2] + data.bearing)
                self.mu[2*landmark_id+2] = self.mu[0] + \
                    data.range * math.sin(self.mu[2] + data.bearing)
                self.observed_landmarks[landmark_id] = True


def main():
    landmarks = read_world()
    timesteps = read_data()
    ekf = EKFSLAM(landmarks, timesteps)
    for i in range(len(timesteps)):
        ekf.prediction(i)
        ekf.correction(i)


if __name__ == "__main__":
    main()
