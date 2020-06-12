from collections import deque


# Historical Averaging
# https://arxiv.org/pdf/1606.03498.pdf
class HistoricalAverageLoss:
    def __init__(self, model, moving_average_length=32):
        super().__init__()
        self.model = model
        self.moving_average_length = moving_average_length
        self.params = deque(maxlen=self.moving_average_length)
        self.timestep = 0

    def tick(self):
        current_params = sum([param.sum() for param in self.model.parameters()])
        summed_params = sum(self.params)

        if self.timestep > 0:
            mean_params = summed_params / self.timestep
            loss = (current_params - mean_params) ** 2
        else:
            loss = 0

        self.params.append(current_params)
        self.timestep = self.timestep + 1 if self.timestep < self.moving_average_length else self.timestep

        return loss
