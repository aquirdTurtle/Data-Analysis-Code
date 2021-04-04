import dataclasses as dc
import numpy as np

@dc.dataclass
class PictureWindow:
    """
    A structure that holds all of the info relevant for determining thresholds.
    """
    # the actual threshold
    xmin:int = 0
    xmax:int = None
    ymin:int = 0
    ymax:int = None
    def __str__(self):
        return ('[Picture window with xmin: ' + str(self.xmin) + ', xmax:'+str(self.xmax)
                +', ymin:'+str(self.ymin)+', ymax:'+str(self.ymax)) + ']'
    def __repr__(self):
        return self.__str__()
    def window(self, pic_s):
        """
        pic_s can be a single pic or an array of pics.
        """
        if len(np.array(pic_s).shape) == 2:
            return pic_s[self.ymin:self.ymax, self.xmin:self.xmax]
        elif len(np.array(pic_s).shape) == 3:
            return pic_s[:,self.ymin:self.ymax, self.xmin:self.xmax]

