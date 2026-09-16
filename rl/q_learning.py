import random
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        n_carrier_actions=6,
        n_placer_actions=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.n_carrier_actions = n_carrier_actions
        self.n_placer_actions = n_placer_actions
        self.n_actions = n_carrier_actions * n_placer_actions  # 12 actions combinées

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # La Q-table : pour chaque état jamais vu, on démarre avec
        # une liste de n_actions scores à 0.0 (defaultdict s'en charge
        # automatiquement dès qu'on accède à un nouvel état).
        self.q_table = defaultdict(lambda: [0.0] * self.n_actions)

    def _state_to_key(self, observation):
        """
        Transforme l'observation (array numpy de floats) en une clé
        hashable (tuple d'entiers) utilisable comme clé de dictionnaire.
        """
        return tuple(int(round(v)) for v in observation)

    def _action_to_index(self, carrier_action, placer_action):
        """Combine les 2 actions séparées en un seul indice 0-11."""
        return carrier_action * self.n_placer_actions + placer_action

    def _index_to_action(self, index):
        """Opération inverse : un indice 0-11 -> (carrier_action, placer_action)."""
        carrier_action = index // self.n_placer_actions
        placer_action = index % self.n_placer_actions
        return carrier_action, placer_action

    def choose_action(self, observation):
        """
        Stratégie epsilon-greedy :
        - avec probabilité epsilon -> action complètement aléatoire (exploration)
        - sinon -> la meilleure action connue pour cet état (exploitation)
        Renvoie directement (carrier_action, placer_action).
        """
        if random.random() < self.epsilon:
            index = random.randrange(self.n_actions)
        else:
            state_key = self._state_to_key(observation)
            q_values = self.q_table[state_key]
            index = q_values.index(max(q_values))

        return self._index_to_action(index)

    def update(self, observation, carrier_action, placer_action, reward, next_observation, done):
        """
        Met à jour la Q-table après avoir vécu une transition
        (state, action, reward, next_state).
        Formule de Bellman : on rapproche le score actuel de
        "reward reçu + estimation du meilleur futur possible".
        """
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)
        action_index = self._action_to_index(carrier_action, placer_action)

        current_q = self.q_table[state_key][action_index]

        if done:
            # Épisode terminé : pas de futur à prendre en compte
            target = reward
        else:
            best_next_q = max(self.q_table[next_state_key])
            target = reward + self.gamma * best_next_q

        # Mise à jour progressive (pas un remplacement brutal)
        self.q_table[state_key][action_index] = current_q + self.lr * (target - current_q)

    def decay_epsilon(self):
        """À appeler une fois par épisode terminé, pour explorer de moins en moins."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)