from abc import ABC

class base_cmd_interface(ABC):

    def __str_to_int(self, value):
        val = 0
        try:
            val = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("not an integer: " + value)
        return val

    def unsigned_int(self, value):
        val = self.__str_to_int(value)
        if val < 0:
            raise argparse.ArgumentTypeError("must be a non-negative integer")
        return val
        

    def positive_int(self, value):
        val = self.__str_to_int(value)
        if val < 1:
            raise argparse.ArgumentTypeError("must be an integer greater than 0")
        return val
   

    def min_dim_size(self, value):
        val = self.__str_to_int(value)
        if val < 28 or val > 48:
            raise argparse.ArgumentTypeError("must be an integer >= 28 and <= 48")
        return val