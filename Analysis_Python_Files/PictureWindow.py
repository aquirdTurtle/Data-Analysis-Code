import dataclasses as dc
import numpy as np
import warnings

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
        """ pic_s can be a single pic or an array of pics.
        """

        picShape = np.array(pic_s).shape if len(np.array(pic_s).shape) == 2 else (np.array(pic_s).shape[1], np.array(pic_s).shape[2])
        egPic = pic_s if len(np.array(pic_s).shape) == 2 else pic_s[0]
        finEgPic = np.array(egPic[self.ymin:self.ymax, self.xmin:self.xmax])
        if (self.ymax is not None and self.ymax > picShape[0]) or (self.xmax is not None and self.xmax > picShape[1]):
            warnings.warn('Warning: Picture window max is larger than picture dimensions! ' + str(self) + ', ' + str(picShape))
        if finEgPic.shape[0] == 0 or finEgPic.shape[1] == 0:
            warnings.warn('Warning: Windows picture size is zero! ' + str(self) + ', ' + str(picShape) + str(finEgPic.shape))
        if len(np.array(pic_s).shape) == 2:
            return pic_s[self.ymin:self.ymax, self.xmin:self.xmax]
        elif len(np.array(pic_s).shape) == 3:
            return pic_s[:,self.ymin:self.ymax, self.xmin:self.xmax]

