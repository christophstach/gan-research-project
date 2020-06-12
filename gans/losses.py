from collections import deque


# Historical Averaging
# https://arxiv.org/pdf/1606.03498.pdf
class HistoricalAverageLoss:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.timesteps = 0
        self.sum_parameters = []
        self.lambd = lambd

    def tick(self):
        if self.timesteps == 0:
            for p in self.model.parameters():
                param = p.data.clone()
                self.sum_parameters.append(param)

            self.timesteps += 1

            return 0.0
        else:
            loss = 0.0
            for i, p in enumerate(self.model.parameters()):
                loss += torch.sum(
                    (p - (self.sum_parameters[i].data / self.timesteps)) ** 2
                )

                self.sum_parameters[i] += p.data.clone()

            self.timesteps += 1
            loss *= self.lambd

            return loss
