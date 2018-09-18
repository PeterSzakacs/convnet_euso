# classes and functions for manipulating data in packets of simulated frames

# Packet data template

class packet_template():

    def __init__(self, EC_width, EC_height, frame_width, frame_height, frames_per_packet):
        self._EC_width, self._EC_height = EC_width, EC_height
        self._width, self._height = frame_width, frame_height
        self._num_rows = int(frame_height/EC_height)
        self._num_cols = int(frame_width/EC_width)
        self._num_EC = self._num_rows * self._num_cols
        self._num_frames = frames_per_packet

    #properties

    @property
    def packet_shape(self):
        return (self._num_frames, self._width, self._height)

    @property
    def frame_width(self):
        return self._width

    @property
    def frame_height(self):
        return self._height

    @property
    def EC_width(self):
        return self._EC_width

    @property
    def EC_height(self):
        return self._EC_height

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_cols(self):
        return self._num_cols

    @property
    def num_EC(self):
        return self._num_EC
    
    @property
    def num_frames(self):
        return self._num_frames

    # unit conversions

    def x_to_ec_x(self, x):
        return int(x/self._EC_width)

    def y_to_ec_y(self, y):
        return int(y/self._EC_height)

    def xy_to_ec_idx(self, x, y):
        return self.ec_xy_to_ec_idx(self.x_to_ec_x(x), self.y_to_ec_y(y))

    def ec_xy_to_ec_idx(self, ec_x, ec_y):
        return ec_x + self._num_cols * ec_y

    def ec_idx_to_ec_xy(self, ec_idx):
        return ec_idx % self._num_cols, int(ec_idx / self._num_cols)