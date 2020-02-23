from utils import read_data, read_world
import numpy as np


class EKFSLAM():
    def __init__(self, landmarks, timesteps):
        self.landmarks = landmarks
        self.timesteps = timesteps
        self.N = len(landmarks)
        self.observed_landmarks = np.zeros(self.N, dtype=bool)
        self.mu = np.zeros((2*self.N+3, 1))
        self.sigma = np.identity(2*self.N+3) * 1e3
        self.sigma[0:3, 0:3] = np.zeros((3, 3))

    def prediction(self):
        pass

    def correction(self):
        pass


def main():
    landmarks = read_world()
    timesteps = read_data()
    ekf = EKFSLAM(landmarks, timesteps)
    print(ekf.sigma)


if __name__ == "__main__":
    main()
