class TTSplitter:
    def __init__(self, samples, targets, st_size=None):
        self.samples = samples
        self.targets = targets
        self.st_size = st_size

    def split(self):
        tt_size = int(self.targets.size(0) * self.st_size)
        st_size = int(self.samples.size(0) * self.st_size)

        return self.samples[:st_size, :], self.samples[st_size:, :],self.targets[:tt_size], self.targets[tt_size:]
