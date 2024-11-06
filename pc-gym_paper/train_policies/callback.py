from stable_baselines3.common.callbacks import BaseCallback
import csv


# Callback to write the reward and timestep to a .csv file while training with SB3
class LearningCurveCallback(BaseCallback):
    def __init__(self, verbose=0, log_file="learning_curve.csv"):
        super(LearningCurveCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.log_file = log_file

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            # Record episode rewards during training
            self.episode_rewards.append(self.model.ep_info_buffer[0].get('r', 0.0))

        return True

    def _on_training_end(self):
        # Save rewards to CSV file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Reward'])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i, reward])