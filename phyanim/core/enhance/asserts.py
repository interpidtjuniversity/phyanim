import numpy as np

class Content:
    def __init__(self, pos_variables: tuple[str, str], obj):
        self.pos_variables = pos_variables
        self.obj = obj
    
    def change_callback(self):
        return lambda x_pos, y_pos: self.update(x_pos, y_pos)
    
    def update(self, x_pos, y_pos):
        self.obj.move_to(np.array([x_pos, y_pos, 0.0]))
