from ctypes import *
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int), ("data",
                                                           POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


class Darknet(object):
    """Wrapper for darknet APIs from dynamic lib"""

    def __init__(self, cfg, meta, weights):
        libdarknet_path = os.path.join(THIS_DIR, "..", "libdarknet.so")
        lib = CDLL(libdarknet_path, RTLD_GLOBAL)
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        self.predict = lib.network_predict_p
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.make_boxes = lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype = POINTER(BOX)

        self.free_ptrs = lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes = lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype = c_int

        self.make_probs = lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype = POINTER(POINTER(c_float))

        self.reset_rnn = lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = lib.load_network_p
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.network_detect = lib.network_detect
        self.network_detect.argtypes = [
            c_void_p, IMAGE, c_float, c_float, c_float,
            POINTER(BOX),
            POINTER(POINTER(c_float))
        ]

        self.network_visual_detect = lib.network_visual_detect
        self.network_visual_detect.argtypes = [
            c_void_p, IMAGE, c_float, c_float, c_float,
            POINTER(BOX),
            POINTER(POINTER(c_float)),
            POINTER(c_char_p),
            POINTER(POINTER(IMAGE)), c_char_p
        ]

        self.draw_detections = lib.draw_detections
        self.draw_detections.argtypes = [
            IMAGE, c_int, c_float,
            POINTER(BOX),
            POINTER(POINTER(c_float)),
            POINTER(POINTER(c_float)),
            POINTER(c_char_p),
            POINTER(POINTER(IMAGE)), c_int, c_int
        ]

        self.load_alphabet = lib.load_alphabet
        self.load_alphabet.argtypes = []
        self.load_alphabet.restype = POINTER(POINTER(IMAGE))

        self.save_image = lib.save_image
        self.save_image.argtypes = [IMAGE, c_char_p]

        self._init_network(cfg, meta, weights)

    def _init_network(self, cfg, meta, weights):
        self.net = self.load_net(cfg, weights, 0)
        self.meta_f = meta
        self.meta = self.load_meta(meta)

    def _get_names(self, meta):
        names_file = None
        with open(meta, 'w') as f:
            for l in f.readlines():
                if l.startswith("names"):
                    names_file = l.split('=')[1].strip()
                    break

        if names_file is None:
            raise Exception("Cannot find 'names' entry in {}".format(meta))

        with open(names_file) as fn:
            names = [i.strip() for i in fn.readlines()]
            return names

    def classify(self, im):
        out = self.predict_image(self.net, im)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45, outfile=None):
        im = self.load_image(image, 0, 0)
        boxes = self.make_boxes(self.net)
        probs = self.make_probs(self.net)
        num = self.num_boxes(self.net)
        self.network_detect(self.net, im, thresh, hier_thresh, nms, boxes,
                            probs)

        # draw bounding boxes
        if outfile is not None:
            masks = POINTER(POINTER(c_float))()
            alphabet = self.load_alphabet()
            names = self.meta.names
            classes = self.meta.classes
            self.draw_detections(im, num, thresh, boxes, probs, masks, names,
                                 alphabet, classes, 1)
            self.save_image(im, outfile)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.names[i], probs[j][i],
                                (boxes[j].x, boxes[j].y, boxes[j].w,
                                 boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res

    def detect_v(self, image, outfile, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        boxes = self.make_boxes(self.net)
        probs = self.make_probs(self.net)
        num = self.num_boxes(self.net)
        alphabet = self.load_alphabet()
        names = self.meta.names

        self.network_visual_detect(self.net, im, thresh, hier_thresh, nms,
                                   boxes, probs, names, alphabet, outfile)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.names[i], probs[j][i],
                                (boxes[j].x, boxes[j].y, boxes[j].w,
                                 boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res


if __name__ == "__main__":
    cfg = os.path.join(THIS_DIR, "..", "cfg", "yolo.cfg")
    meta = os.path.join(THIS_DIR, "..", "cfg", "coco.data")
    weights = os.path.join(THIS_DIR, "..", "yolo.weights")
    img = os.path.join(THIS_DIR, "..", "data", "dog.jpg")

    yolo = Darknet(cfg, meta, weights)
    # res = yolo.detect(img, outfile="yolo_det")
    res = yolo.detect_v(img, outfile="yolo_det")
    print(res)
