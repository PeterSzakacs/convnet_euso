# generator functions for shower line values

class default_vals_generator():

    def __init__(self, maximum, duration):
        self.reset(maximum, duration)

    def reset(self, maximum, duration):
        self.duration, self.maximum = duration, maximum
        self.iteration = 0
        self.maxinv = 1/maximum

    def __iter__(self):
        return self

    def __next__(self):
        if (self.iteration < self.duration):
            self.iteration += 1
            return round(-self.maxinv * pow(self.iteration - 2, 2) + self.maximum)
        else:
            raise StopIteration()

class flat_vals_generator():

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

# class to perform various operations on packet data (create packet projections,
# zero-out EC cells, draw a simulated shower line with a generator function, ...)