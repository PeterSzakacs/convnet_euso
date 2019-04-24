# generator functions for shower line values


class DefaultValsGenerator():

    def __init__(self, maximum, duration):
        self.reset(maximum, duration)

    def reset(self, maximum, duration):
        self.duration, self.max = duration, maximum
        self.iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.duration):
            self.iteration += 1
            return round(
                self.max * (-pow(2*self.iteration / self.duration - 1, 2) + 1)
            )
        else:
            raise StopIteration()


class FlatValsGenerator():

    def __init__(self, maximum, duration):
        self.reset(maximum, duration)

    def reset(self, maximum, duration):
        self.iteration = 0
        self.duration = duration
        self.maximum = maximum

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.duration):
            self.iteration += 1
            return self.maximum
        else:
            raise StopIteration()
