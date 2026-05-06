from typing import Any, TypeVar, SupportsFloat

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from .reward_functions import Gaussian, Polynomial

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BoundlessBuffaloEnv(gym.Env):
    def __draw_polynomial(self):
        """
        Draw a new set of coefficients for the reward polynomial
        """
        self.rng = np.random.default_rng(self.seed)
        coefficients = [self.rng.uniform(-self.coef_range, self.coef_range) for _ in range(self.degree + 1)]
        coefficients[0] = 0
        if coefficients[-1] > 0:
            coefficients[-1] *= -1

        self.polynomial = np.polynomial.Polynomial(coefficients)
        d1: np.polynomial.Polynomial = self.polynomial.deriv()
        d2 = d1.deriv()

        roots = [root.real for root in d1.roots() if np.isrealobj(root) and d2(root.real) < 0]
        maximum = max([self.polynomial(root) for root in roots], default=0)
        self.polynomial.coef[0] += self.max_val - maximum
        self.reward_model = Polynomial(coefficients=self.polynomial.coef)

        self.left_shoulder = -np.inf
        self.right_shoulder = np.inf
        if self.shoulders:
            roots = [root.real for root in d1.roots() if np.isrealobj(root)]
            minimum = min([self.polynomial(root) for root in roots], default=0)
            minimum -= abs(minimum)
            cross = np.polynomial.Polynomial(self.polynomial.coef)
            cross.coef[0] += -minimum
            shoulders = [root.real for root in cross.roots()]
            self.left_shoulder = min(shoulders, default=-np.inf)
            self.right_shoulder = max(shoulders, default=np.inf)
        self.coefficient = self.polynomial.coef

    def __draw_predefined_polynomial(self, polynomial: int):
        """
        Loads a predefined polynomial based on the input integer
        :param polynomial: integer which determines which predefined polynomial to load
        """
        self.coefficient = 1
        if polynomial == 1:
            self.polynomial = lambda x: (np.exp(-40 * (x - 0.35) ** 2) + np.exp(-40 * (x - 0.65) ** 2))/1.31
            self.reward_model = Gaussian(mus=[0.35, 0.65], alphas=[40.0, 40.0], coefs=[1.0, 1.0], norm=1.31)
            self.left_shoulder = 0.0
            self.right_shoulder = 1.0
        elif polynomial == 2 or polynomial == 3:
            if polynomial == 2:
                coefficients = [1.1, -2.9, 3, 7.3, -1.4, -1.5, 2.3, -2.8, -2.7]
                poly_coefficients = coefficients
                powers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            elif polynomial == 3:
                coefficients = [0.1, 0.4, -0.08]
                poly_coefficients = [0.1, 0, 0, 0, 0, 0, 0.4, 0, 0, 0, -0.08]
                powers = [0, 6, 10]
            self.polynomial = np.polynomial.Polynomial(poly_coefficients)
            self.reward_model = Polynomial(coefficients, powers)
            roots = [root.real for root in self.polynomial.roots()]
            self.left_shoulder = min(roots)
            self.right_shoulder = max(roots)
        elif polynomial == 4:
            self.polynomial = lambda x: (0.5 * np.exp(-100 * (x - 0.6) ** 2) + 0.5 * np.exp(-2 * (x - 1.4) ** 2))
            self.reward_model = Gaussian(mus=[0.6, 1.4], alphas=[100.0, 2.0], coefs=[0.5, 0.5], norm=1)
            self.left_shoulder = 0.0
            self.right_shoulder = 3.0
        elif polynomial == 5:
            self.polynomial = lambda x: (0.41 * np.exp(-80 * (x - 0.2) ** 2) + 0.37 * np.exp(-60 * (x - 0.4) ** 2) + 0.4 * np.exp(-80 * (x - 0.6) ** 2) + 0.3 * np.exp(-50 * (x - 0.8) ** 2))
            self.reward_model = Gaussian(mus=[0.2, 0.4, 0.6, 0.8], alphas=[80.0, 60.0, 80.0, 50.0], coefs=[0.41, 0.37, 0.4, 0.3], norm=1)
            self.left_shoulder = 0.0
            self.right_shoulder = 1.0
        else:
            raise ValueError("'predefined_polynomial' must be 1, 2, 3, 4, 5, or None")

    def __init__(
        self,
        degree: int = 2,
        dynamic_rate: int | None = None,
        seed: int | None = None,
        std_deviation: float = 0.1,
        coef_range: float = 10,
        max_val: float = 10.0,
        shoulders: bool = True,
        shoulder_leakage: float = 0.0,
        predefined_polynomial: int | None = None,
        binary_reward: bool = False,
    ):
        """
        Infinite armed bandit environment.  The input is scaled from (-inf, +inf) to (-1, +1) in an attempt to keep
        this numerically stable.  Also, coefficients are drawn from (-0.1, 0.1) to help this along.
        :param degree: Degree of polynomial which defines the reward function
        :param dynamic_rate: number of pulls between drawing a new polynomial, NONE if not dynamic
        :param seed: Randomness seed, NONE if it doesn't matter
        :param std_deviation: randomness around reward function
        :param predefined_polynomial: Integer value to load predefined function for reward
        :param binary_reward: Determines if reward is deterministic or sampled from reward probability function
        """
        if degree < 2 or degree % 2 == 1:
            raise ValueError("degree must be an even number greater than or equal to 2")

        self.initial_seed = seed
        self.seed = seed
        self.degree = degree
        self.dynamic_rate = dynamic_rate
        self.std_deviation = std_deviation
        self.coef_range = coef_range
        self.max_val = max_val
        self.shoulders = shoulders
        self.shoulder_leakage = shoulder_leakage
        self.binary_reward = binary_reward

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        if not predefined_polynomial:
            self.__draw_polynomial()
        elif predefined_polynomial:
            self.__draw_predefined_polynomial(predefined_polynomial)
        self.pulls = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """ "
        Resets the environment
        :param seed: WARN unused, defaults to None
        :param options: WARN unused, defaults to None
        :return: observation, info
        """
        self.seed = seed
        self.pulls = 0

        return np.zeros((1,), dtype=np.float32), {"coef": self.coefficient}

    def step(self, action: float) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Steps the environment
        :param action: One of infinite arms to pull in (-inf, +inf)
        :return: observation, reward, done, term, info
        """
        if self.shoulders and action < self.left_shoulder:
            left = self.polynomial(self.left_shoulder)
            reward = left + self.shoulder_leakage * (self.polynomial(action)[0] - left)
        elif self.shoulders and action > self.right_shoulder:
            right = self.polynomial(self.right_shoulder)
            reward = right + self.shoulder_leakage * (self.polynomial(action)[0] - right)
        else:
            reward = self.polynomial(action)[0]
        if self.binary_reward:
            reward = 1 if np.random.rand() <= reward else 0

        self.pulls += 1  # Fixed double increment bug
        if self.dynamic_rate is not None and self.pulls % self.dynamic_rate == 0:
            if self.seed is not None:
                self.seed += 1
            self.__draw_polynomial()

        return np.zeros((1,), dtype=np.float32), reward, False, False, {"coef": self.coefficient}

    def plot_polynomial(self):
        x = np.linspace(self.left_shoulder, self.right_shoulder, 1000)
        y = self.polynomial(x)
        plt.plot(x, y)
        plt.xlabel("Action")
        plt.ylabel("Reward")
        plt.title("Reward Polynomial")
        plt.grid()
        plt.show()
