import os
import pytlsd
import numpy as np
import cv2

from collections import namedtuple
BaseDetectorOptions = namedtuple("BaseDetectorOptions",
                                 ["set_gray", "max_num_2d_segs", "do_merge_lines", "visualize", "weight_path"],
                                 defaults=[True, 3000, False, False, None])

class LSDDetector():
    def __init__(self, options = BaseDetectorOptions()):
        self.set_gray = options.set_gray
        self.max_num_2d_segs = options.max_num_2d_segs

    def get_module_name(self):
        return "lsd"

    def detect(self, ori_img):
        if self.set_gray:
            img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            img = ori_img
        segs = pytlsd.lsd(img.astype(np.float64))
        return segs

    def detect_with_min_length(self, ori_img, min_length):
        if self.set_gray:
            img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            img = ori_img
        segs = pytlsd.lsd(img * 255)
        start_point = segs[:, 0:2]
        end_point = segs[:, 2:4]
        line_length = end_point - start_point
        valid_mask = np.linalg.norm(line_length, axis=1) > min_length
        seg_filter = segs[valid_mask, :]
        return seg_filter

    def vis_extract_line(self, ori_img):
        if self.set_gray:
            img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            img = ori_img
        cv2.imshow("Gray img", img)
        segs = pytlsd.lsd(img*255)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for seg in segs:
            cv2.line(img_color, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), (0, 255, 0))

        print(f"Detected segments N {len(segs)}")
        cv2.imshow("Extract line", cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        cv2.waitKey()

    def take_longest_k(self, segs, max_num_2d_segs=3000):
        indexes = np.arange(0, segs.shape[0])
        if max_num_2d_segs is None or max_num_2d_segs == -1:
            pass
        elif segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:, 2] - segs[:, 0]) ** 2 + (segs[:, 3] - segs[:, 1]) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes, :]
        return segs, indexes

