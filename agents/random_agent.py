from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    name = "random"

    def act(self, obs):
        return self.rng.choice(obs.available_actions)
