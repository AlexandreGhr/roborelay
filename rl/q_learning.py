"""
Tabular Q-learning agent.

The Q-table gives, for each state the agent has seen, a score for each action:
"how much reward can I expect if I do this action here, then play well".
One agent controls both robots, so an action is a pair (carrier_action, placer_action),
stored as a single number from 0 to 11.
"""

import pickle
import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        n_carrier_actions=6,
        n_placer_actions=2,
        learning_rate=0.1,      # alpha: how big each update step is
        discount_factor=0.95,   # gamma: how much future rewards count
        epsilon=1.0,            # chance of a random action (exploration)
        epsilon_decay=0.999,    # epsilon is multiplied by this after each game
        epsilon_min=0.1,        # epsilon never goes below this during training
        seed=None,
    ):
        self.n_carrier_actions = n_carrier_actions
        self.n_placer_actions = n_placer_actions
        self.n_actions = n_carrier_actions * n_placer_actions  # 12 combined actions

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Own random generator: the same seed gives the exact same training
        self.rng = random.Random(seed)

        # Q-table: state -> list of n_actions scores.
        # A state never seen before starts with all scores at 0.0 (defaultdict does it).
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)

    # --- Conversions -----------------------------------------------------

    def _state_to_key(self, observation):
        """A numpy array cannot be a dict key: turn the observation into a tuple of ints."""
        return tuple(int(round(v)) for v in observation)

    def _action_to_index(self, carrier_action, placer_action):
        """Pair of actions -> one number from 0 to 11."""
        return carrier_action * self.n_placer_actions + placer_action

    def _index_to_action(self, index):
        """One number from 0 to 11 -> pair of actions."""
        return index // self.n_placer_actions, index % self.n_placer_actions

    # --- Acting and learning ---------------------------------------------

    def choose_action(self, observation):
        """Epsilon-greedy: random action with probability epsilon, otherwise the best known one.

        Returns (carrier_action, placer_action).
        """
        if self.rng.random() < self.epsilon:
            index = self.rng.randrange(self.n_actions)
        else:
            q_values = self.q_table[self._state_to_key(observation)]
            best = max(q_values)
            # Break ties at random: otherwise, in a state never seen (all scores 0),
            # the agent would always pick action 0.
            best_actions = [i for i, q in enumerate(q_values) if q == best]
            index = self.rng.choice(best_actions)

        return self._index_to_action(index)

    def update(self, observation, carrier_action, placer_action, reward, next_observation, terminated):
        """Bellman update after one step:

            Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
            target   = reward + gamma * max Q(s', .)     (or just reward if terminated)

        terminated must be True only when the game really ended (item stored),
        NOT when it was cut by the turn limit (truncated): after a time limit
        the future still exists, the agent just ran out of time.
        """
        state = self._state_to_key(observation)
        next_state = self._state_to_key(next_observation)
        action = self._action_to_index(carrier_action, placer_action)

        if terminated:
            target = reward  # real end: nothing more to win
        else:
            target = reward + self.gamma * max(self.q_table[next_state])

        current = self.q_table[state][action]
        # Move the score a small step towards the target (not a full replacement)
        self.q_table[state][action] = current + self.lr * (target - current)

    def decay_epsilon(self):
        """Call once after each game: explore a bit less each time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --- Save / load -----------------------------------------------------

    def save(self, path):
        """Save the Q-table and the settings used to train it."""
        data = {
            "q_table": dict(self.q_table),  # defaultdict -> dict (pickle cannot save the lambda)
            "n_carrier_actions": self.n_carrier_actions,
            "n_placer_actions": self.n_placer_actions,
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, seed=None):
        """Rebuild a trained agent from a file, with no exploration (epsilon = 0).

        Only load files you created yourself: a pickle file can run code when opened.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            n_carrier_actions=data["n_carrier_actions"],
            n_placer_actions=data["n_placer_actions"],
            learning_rate=data.get("learning_rate", 0.1),
            discount_factor=data.get("discount_factor", 0.95),
            epsilon=0.0,
            seed=seed,
        )
        agent.q_table.update(data["q_table"])
        return agent