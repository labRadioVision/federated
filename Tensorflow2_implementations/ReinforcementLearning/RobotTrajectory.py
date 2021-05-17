import mat73
import numpy as np

class RobotTrajectory:
    def __init__(self, filepath, lookuptab, rewards, file_size, number_positions):
        self.filepath = filepath
        self.lookuptab_file = lookuptab
        self.rewardsfile = rewards
        s_data = mat73.loadmat(self.filepath)
        tab_l = mat73.loadmat(self.lookuptab_file)
        rew = mat73.loadmat(self.rewardsfile)
        self.data_test = s_data['data_hexapods']
        self.lookuptab = tab_l['lookuptab']
        self.rewards_vec = rew['rewards']
        self.number_devices = 5
        self.number_positions = number_positions
        self.done = False

    def initialize(self, position_initial=0, random=False):
        # fixed start
        if not random:
            self.initial_position = position_initial
            self.position = position_initial
            frame_sample = int(np.random.choice(np.arange(self.number_devices)))
            self.obs = self.data_test[self.position][frame_sample]["camera"]
            self.rewards = self.rewards_vec[self.position, 1]  # change
            self.done = self.data_test[self.position][frame_sample]["done"]
        else:
            print("error, only non random initialization supported")
        return self.obs, self.rewards, self.done
    # implement the action on the device, return observation and rewards
    def implement(self, action, device):
        # next position given the action
        if device < self.number_devices:
            self.position = int(self.lookuptab[self.position, action]) # find the new position
            self.rewards = self.rewards_vec[self.position, 1]  # change
            # choose a frame from dataset to select the observation sample
            frame_sample = int(np.random.choice(np.arange(self.number_devices)))
            if self.data_test[self.position][frame_sample]["position"] == self.position:
                self.obs = self.data_test[self.position][frame_sample]["camera"]
                self.done = self.data_test[self.position][frame_sample]["done"]
            else:
                print("wrong sample")
        else:
            print("error, too many devices")
        return self.obs, self.rewards, self.done

    def getPosition(self):
        return self.position