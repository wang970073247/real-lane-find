from collections import deque

class Line():
    def __init__(self):
        self.__frame_num = 5
        self.detected = False
        #self.bestx = None
        self.best_fit = None
        #self.line_base_pos = None
        self.lane_inds = []
        self.allx = None
        self.ally = None
        self.radius_of_curvature = None
        #self.line_base_pos = None
        #self.bottom_x_pos = deque(maxlen=self.__frame_num)
        self.recent_xfitted = deque(maxlen=self.__frame_num)
