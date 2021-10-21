import cv2
import numpy as np
import glob
import os
import tempfile
from collections import OrderedDict
from tqdm import tqdm
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.structures.image_list import to_image_list
import matplotlib.pyplot as plt

from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                 CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

from mega_core.data.datasets.vid import VIDDataset
import sys
import matplotlib.pyplot
if sys.version_info[0]==2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from tools.utils import convert_BoxList_to_Box, convert_vid_annotations_to_Box
from analyze.analyzer import Analyzer
from analyze.visualize import normalize_image_for_display

class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoProcessor(object):
    def __init__(self, filename, cache_capacity=10):
        if filename is None:
            self._fps = 25
            self._only_output = True
        else:
            self._vcap = cv2.VideoCapture(filename)
            assert cache_capacity > 0
            self._cache = Cache(cache_capacity)
            self._position = 0
            # get basic info
            self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
            self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
            self._fps = self._vcap.get(CAP_PROP_FPS)
            self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
            self._fourcc = self._vcap.get(CAP_PROP_FOURCC)
            self._only_output = False
        self._output_video_name = "visualization.avi"

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                '"frame_id" must be between 0 and {}'.format(self._frame_cnt -
                                                             1))
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
                return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:07d}.jpg',
                   start=0,
                   max_num=0):
        """Convert a video to frame images

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
        """
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        for i in range(task_num):
            img = self.read()
            if img is None:
                break
            filename = os.path.join(frame_dir,
                                filename_tmpl.format(i + file_start))
            cv2.imwrite(filename, img)

    def frames2videos(self, frames, output_folder):
        if self._only_output:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height, width = frames[0].shape[:2]
        else:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height = self._height
            width = self._width

        videoWriter = cv2.VideoWriter(os.path.join(output_folder, self._output_video_name), fourcc, self._fps, (width, height))

        for frame_id in range(len(frames)):
            videoWriter.write(frames[frame_id])
        videoWriter.release()

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class VIDDemo(object):
    '''
    CATEGORIES = ['__background__',  # always index 0
                  'airplane', 'antelope', 'bear', 'bicycle',
                  'bird', 'bus', 'car', 'cattle',
                  'dog', 'domestic_cat', 'elephant', 'fox',
                  'giant_panda', 'hamster', 'horse', 'lion',
                  'lizard', 'monkey', 'motorcycle', 'rabbit',
                  'red_panda', 'sheep', 'snake', 'squirrel',
                  'tiger', 'train', 'turtle', 'watercraft',
                  'whale', 'zebra']
    '''

    CATEGORIES = ['ignored-regions',
                  'pedestrian','people','bicycle','car',
                  'van','truck','tricycle','awning-tricycle',
                  'bus','motor','others']

    def __init__(
            self,
            cfg,
            method="base",
            confidence_threshold=0.7,
            output_folder="demo/visulaization"
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.method = method
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # for video output
        self.vprocessor = VideoProcessor(None)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_pil_transform(self):
        """
        Creates a basic transformation that was used in generalized_rnn_{}._forward_test()
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def perform_transform(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        return image_list

    # def run_on_image_folder(self, image_folder, suffix='.JPEG'):
    def run_on_image_folder(self, image_folder, suffix='.jpg'):
        image_names = glob.glob(image_folder + '/*' + suffix)
        image_names = sorted(image_names)

        img_dir = "%s" + suffix
        frame_seg_len = len(image_names)
        pattern = image_folder + "/%07d"

        images_with_boxes = []

        for frame_id in tqdm(range(frame_seg_len-1)):
            original_image = cv2.imread(image_names[frame_id])
            img_cur = self.perform_transform(original_image)
            if self.method == "base":
                image_with_boxes = self.run_on_image(original_image, img_cur)
                images_with_boxes.append(image_with_boxes)
            elif self.method in ("dff", "fgfa", "rdn", "mega"):
                infos = {}
                infos["cur"] = img_cur
                infos["frame_category"] = 0 if frame_id == 0 else 1
                infos["seg_len"] = frame_seg_len
                infos["pattern"] = pattern
                infos["img_dir"] = img_dir
                infos["transforms"] = self.build_pil_transform()
                if self.method == "dff":
                    infos["is_key_frame"] = True if frame_id % 10 == 0 else False
                elif self.method in ("fgfa", "rdn"):
                    img_refs = []
                    if self.method == "fgfa":
                        max_offset = self.cfg.MODEL.VID.FGFA.MAX_OFFSET
                    else:
                        max_offset = self.cfg.MODEL.VID.RDN.MAX_OFFSET
                    ref_id = min(frame_seg_len - 1, frame_id + max_offset)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs.append(img_ref)

                    infos["ref"] = img_refs
                elif self.method == "mega":
                    img_refs_l = []
                    # reading other images of the queue (not necessary to be the last one, but last one here)
                    ref_id = min(frame_seg_len - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs_l.append(img_ref)

                    img_refs_g = []
                    if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
                        shuffled_index = np.arange(frame_seg_len)
                        if self.cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        for id in range(size):
                            filename = pattern % shuffled_index[
                                (frame_id + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % frame_seg_len]
                            img = cv2.imread(img_dir % filename)
                            img = self.perform_transform(img)
                            img_refs_g.append(img)

                    infos["ref_l"] = img_refs_l
                    infos["ref_g"] = img_refs_g
                else:
                    pass

                # run inference on current image
                image_with_boxes = self.run_on_image(original_image, infos)
                images_with_boxes.append(image_with_boxes)

                # save image with predicted bboxes
                cv2.imwrite(os.path.join(self.output_folder, "%07d.jpg" % frame_id), image_with_boxes)

            else:
                raise NotImplementedError("method {} is not implemented.".format(self.method))

        return images_with_boxes

    # def run_on_image_folder_VID(self, image_folder, suffix='.JPEG', max_num_frames=None):
    def run_on_image_folder_VID(self, image_folder, suffix='.jpg', max_num_frames=None):

        # initialize analyzer
        self.analyzer = Analyzer(output_dir=self.output_folder,
                                 output_video_name='video.avi',
                                 class_names=VIDDataset.classes,
                                 bbox_match_method='pred_bbox_center',
                                 score_th=self.confidence_threshold
                                 )

        # initialize RPN_analyzer
        rpn_classes = ['object']
        rpn_out_path = os.path.join(self.output_folder, 'RPN')
        clear_out_path = os.path.join(self.output_folder, 'CLEAR')
        self.analyzer_rpn = Analyzer(output_dir=rpn_out_path,
                                     output_video_name='video_rpn.avi',
                                     class_names=rpn_classes,
                                     bbox_match_method='pred_bbox_center',
                                     score_th=0.9
                                     )

        output_image_dir = os.path.join(self.output_folder, 'images')
        output_rpn_image_dir = os.path.join(rpn_out_path, 'images')
        output_clear_image_dir = os.path.join(clear_out_path, 'images')
        os.makedirs(output_image_dir, exist_ok=True)

        #

        image_names = glob.glob(image_folder + '/*' + suffix)
        image_names = sorted(image_names)

        img_dir = "%s" + suffix
        frame_seg_len = len(image_names)
        pattern_image_name = "/%07d"
        pattern = image_folder + pattern_image_name

        # image folder = /home/adiyair/mega.pytorch/datasets/ILSVRC2015_VIS/Data/VID/train_/*

        # annotation folder
        # anno_folder = os.path.join(image_folder, 'annotations')
        anno_folder = image_folder.replace("/Data/","/Annotations/")

        if os.path.exists(anno_folder):
            pattern_anno = anno_folder + "/%07d" + '.xml'
        else:
            anno_folder = None

        # maximum number of frames to be processed
        if max_num_frames >= 0:
            frame_seg_len = np.minimum(frame_seg_len, max_num_frames)

        images_with_boxes = []
        images_rpn_boxes = []
        images_clear_boxes = []
        fig, ax = plt.subplots()
        for frame_id in tqdm(range(frame_seg_len-1)):

            # get image
            original_image = cv2.imread(image_names[frame_id])
            img_cur = self.perform_transform(original_image)

            # run inference
            if self.method == "base":
                image_with_boxes = self.run_on_image(original_image, img_cur)
                images_with_boxes.append(image_with_boxes)

            elif self.method in ("dff", "fgfa", "rdn", "mega"):
                infos = {}
                infos["cur"] = img_cur
                infos["frame_category"] = 0 if frame_id == 0 else 1
                infos["seg_len"] = frame_seg_len
                infos["pattern"] = pattern
                infos["img_dir"] = img_dir
                infos["transforms"] = self.build_pil_transform()
                if self.method == "dff":
                    infos["is_key_frame"] = True if frame_id % 10 == 0 else False
                elif self.method in ("fgfa", "rdn"):
                    img_refs = []
                    if self.method == "fgfa":
                        max_offset = self.cfg.MODEL.VID.FGFA.MAX_OFFSET
                    else:
                        max_offset = self.cfg.MODEL.VID.RDN.MAX_OFFSET
                    ref_id = min(frame_seg_len - 1, frame_id + max_offset)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs.append(img_ref)

                    infos["ref"] = img_refs
                elif self.method == "mega":
                    img_refs_l = []
                    # reading other images of the queue (not necessary to be the last one, but last one here)
                    ref_id = min(frame_seg_len - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs_l.append(img_ref)

                    img_refs_g = []
                    if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
                        shuffled_index = np.arange(frame_seg_len)
                        if self.cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        for id in range(size):
                            filename = pattern % shuffled_index[
                                (frame_id + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % frame_seg_len]
                            img = cv2.imread(img_dir % filename)
                            img = self.perform_transform(img)
                            img_refs_g.append(img)

                    infos["ref_l"] = img_refs_l
                    infos["ref_g"] = img_refs_g
                else:
                    pass

                # run inference on current image
                # predictions, predictions_rpn = self.run_on_image(original_image, infos)[1]
                _, predictions, _, pred_rpn, pca_components = self.run_on_image(original_image, infos)

                # load annotations
                if anno_folder is not None:
                    anno_file = pattern_anno % (frame_id+1)
                    tree = ET.parse(anno_file).getroot()
                    anno = self.preprocess_annotation_vid(tree)

                # save data for analysis
                img_for_analyzer = img_cur  #original_image
                img_for_analyzer_rpn = img_cur
                image_name = image_names[frame_id]
                image_path = os.path.abspath(image_name)
                # print(pred_rpn)
                predictions, pred_rpn, anno, anno_rpn,  image_analyzer,  image_analyzer_rpn  = self.preprocess_data_for_analyzer(predictions, pred_rpn, img_for_analyzer, img_for_analyzer_rpn, anno_type='vid', anno=anno)
                self.analyzer.update_analyzer(key=frame_id, prediction=predictions, ground_truth=anno, image_path=image_path, image=None, analyze_performance=True)
                self.analyzer_rpn.update_analyzer(key=frame_id, prediction=pred_rpn, ground_truth=anno_rpn, image_path=image_path, image=None, analyze_performance=True)
                # plot pca
                if (frame_id >50):
                    x = pca_components[:,0]
                    y = pca_components[:,1]
                    dict = {"car": "red", "people": "blue", "bicycle": "green", "others": "black", "ignored-regions": "black"}
                    tmp =predictions.get_field("labels")
                    color_list =[]
                    from analyze.bounding_box import Box
                    iou = Box.boxes_iou(anno, predictions)
                    gt_box_idx_for_predictions = np.argmax(iou, axis=0)
                    gt_for_predictions = np.array(anno.get_field("labels"))[gt_box_idx_for_predictions]
                    for i in gt_for_predictions:
                        color_list.append(dict[i])
                    plt_obg=[]
                    for obj_class in np.array(["car", "people", "bicycle"]):
                        keep= gt_for_predictions==obj_class
                        plt_obg.append(ax.scatter(x[keep], y[keep], color=dict[obj_class]))
                        # scat=ax.scatter(x[keep], y[keep], color=dict[obj_class],lable=obj_class,title= "Ground truth class")


                    ax.legend(plt_obg, np.array(["car", "people", "bicycle"]),loc="uper right", title="Ground truth")
                    fig.suptitle('PCA- 19 frames fgfa net / inference no video', fontsize=16)
                    # ax.add_artist(legend1)
                    ax.grid(True)
                    # if (frame_id==34):
                    #     plt.show()
                    plt.savefig("/home/adiyair/mega.pytorch/datasets/ILSVRC2015_VIS/ODEval/output/adi_fixed_try/19f_1/"+str(frame_id)+".png")
                # prepare image for video
                image_out_name = os.path.basename(pattern_image_name % frame_id + '.jpg') #'.JPEG')  #'.png')
                image_out_path = os.path.join(output_image_dir, image_out_name)
                image_out_rpn = os.path.join(output_rpn_image_dir, image_out_name)
                image_out_clear = os.path.join(output_clear_image_dir, image_out_name)


                image_rpn_boxes = self.analyzer_rpn.visualize_example(key=frame_id,
                                                                      image=image_analyzer_rpn,
                                                                      show_predictions=True,
                                                                      show_ground_truth=False,
                                                                      rgb2bgr=False,
                                                                      display=False,
                                                                      save_fig_path=image_out_rpn,
                                                                      show_classnames=False,
                                                                      )

                # save image with box to output folder:
                image_with_boxes = self.analyzer.visualize_example(key=frame_id,
                                                                   image=image_analyzer,
                                                                   show_predictions=True,
                                                                   show_ground_truth=True,
                                                                   rgb2bgr=False,
                                                                   display=False,
                                                                   save_fig_path=image_out_path,
                                                                   )

                image_clear_boxes = self.analyzer.visualize_example(key=frame_id,
                                                                    image=image_analyzer,
                                                                    show_predictions=True,
                                                                    show_ground_truth=True,
                                                                    rgb2bgr=False,
                                                                    display=False,
                                                                    save_fig_path=image_out_clear,
                                                                    show_classnames=False,
                                                                    )



                images_with_boxes.append(image_with_boxes)
                images_rpn_boxes.append(image_rpn_boxes)
                images_clear_boxes.append(image_clear_boxes)

            else:
                raise NotImplementedError("method {} is not implemented.".format(self.method))

            # periodically save analyzer and video
            if np.mod(frame_id, 50) == 0:
                self.analyzer.save()
                self.analyzer.update_video(images_with_boxes)
                self.analyzer.update_video(images_clear_boxes)

                self.analyzer_rpn.save()
                self.analyzer_rpn.update_video(images_rpn_boxes)

        # analysis
        self.analyzer.save()
        self.analyzer.update_video(images_with_boxes)
        self.analyzer.update_video(images_clear_boxes)

        self.analyzer_rpn.save()
        self.analyzer_rpn.update_video(images_rpn_boxes)

        self.analyzer.evaluate_performance(generate_report=True)
        self.analyzer_rpn.evaluate_performance(generate_report=True)
        # ax.legend()
        # ax.grid(True)
        # plt.show()
        return images_with_boxes


    def preprocess_data_for_analyzer(self, predictions, pred_rpn, img, img_rpn,  anno_type='vid', anno=None):

        # convert image
        if not isinstance(img, np.ndarray):  # if img is a tensor - convert to ndarray
            img = img.tensors.squeeze().to(self.cpu_device)
        img = normalize_image_for_display(img, bgr2rgb=True)

        if not isinstance(img_rpn, np.ndarray):  # if img is a tensor - convert to ndarray
            img_rpn = img_rpn.tensors.squeeze().to(self.cpu_device)
        img_rpn = normalize_image_for_display(img_rpn, bgr2rgb=True)

        image_shape = img.shape


        predictions = convert_BoxList_to_Box(predictions)
        predictions.resize(image_shape)


        pred_rpn = convert_BoxList_to_Box(pred_rpn)
        if len(pred_rpn.bbox) > 0:
            pred_rpn.resize(image_shape)
        else:
            print("skip")


        try:
            if anno_type == 'vid':
                anno, anno_rpn = convert_vid_annotations_to_Box(anno)
                anno.resize(image_shape)
                anno_rpn.resize(image_shape)

        except:
            anno = None
            anno_rpn = None

        return predictions, pred_rpn, anno, anno_rpn, img, img_rpn


    def preprocess_annotation_vid(self, target):

        classes_map = VIDDataset.classes_map
        classes_to_ind = dict(zip(classes_map, range(len(classes_map))))
        ind_to_class_names = {ind: VIDDataset.classes[ind] for ind in range(len(VIDDataset.classes))}

        boxes = []
        gt_classes = []

        size = target.find('size')
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))

        objs = target.findall('object')
        for obj in objs:
            if not obj.find('name').text in classes_map:
                continue

            bbox = obj.find('bndbox')
            box = [
                np.maximum(float(bbox.find('xmin').text), 0),
                np.maximum(float(bbox.find('ymin').text), 0),
                np.minimum(float(bbox.find('xmax').text), im_info[1] -1),
                np.minimum(float(bbox.find('ymax').text), im_info[0] -1),
            ]

            boxes.append(box)

            class_ind = classes_to_ind[obj.find('name').text.lower().strip()]
            target_class = ind_to_class_names[class_ind]

            gt_classes.append(target_class)

        res = {
            'boxes': np.array(boxes, dtype=np.float32).reshape(-1, 4),
            'labels': gt_classes,
            'im_info': im_info
        }

        return res


    def run_on_video(self, video_path):
        if not os.path.isfile(video_path):
            raise FileNotFoundError('file "{}" does not exist'.format(video_path))
        self.vprocessor = VideoProcessor(video_path)
        tmpdir = tempfile.mkdtemp()
        self.vprocessor.cvt2frames(tmpdir)
        results = self.run_on_image_folder(tmpdir, suffix='.jpg')

        return results

    def run_on_image(self, image, infos=None):
        """
        Arguments:
            image
            infos
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions, predictions_RPN, pca_componennts = self.compute_prediction(image, infos)
        top_predictions, pca_componennts_top = self.select_top_predictions(predictions,pca_componennts)
        top_RPN_predictions = self.select_top_RPN_predictions(predictions_RPN)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        result_rpn = image.copy()
        # result_rpn = self.overlay_boxes(result_rpn, top_RPN_predictions)

        return result, top_predictions, result_rpn, top_RPN_predictions, pca_componennts_top
        # return result, top_predictions


    def compute_prediction(self, original_image, infos):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # compute predictions
        with torch.no_grad():
            predictions, predictions_rpn = self.model(infos)
        predictions = [o.to(self.cpu_device) for o in predictions]
        predictions_rpn = [o.to(self.cpu_device) for o in predictions_rpn]
        # always single image is passed at a time


        prediction = predictions[0]
        prediction_rpn = predictions_rpn[0]

        from sklearn.decomposition import PCA

        tmp_vector = prediction.get_field("vector")

        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(tmp_vector)
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(1, 1, 1)
        # x=pca_components[:,0]
        # y=pca_components[:,1]
        # ax= plt.plot(x,y,"o")
        # for i, txt in enumerate(prediction.get_field("labels")):
        #     ax.annotate(txt, (x[i], y[i]))
        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        prediction_rpn = prediction_rpn.resize((width, height))


        # scores_rpn = prediction_rpn.get_field("objectness")
        # labels_rpn = scores_rpn/scores_rpn
        # prediction_rpn.add_field("scores", scores_rpn)
        # prediction_rpn.add_field("labels", labels_rpn)




        return prediction, prediction_rpn,pca_components

    def select_top_predictions(self, predictions, pca_componnets):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        pca_componnets = pca_componnets[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx], pca_componnets[idx]

    def select_top_RPN_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        objectness = predictions.get_field("objectness")
        keep = torch.nonzero(objectness > 0.95).squeeze(1)
        predictions = predictions[keep]
        objectness = predictions.get_field("objectness")
        _, idx = objectness.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        labels = predictions.get_field("labels")
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        return image

    def generate_images(self, visualization_results):
        for frame_id in range(len(visualization_results)):
            cv2.imwrite(os.path.join(self.output_folder, "%07d.jpg" % frame_id), visualization_results[frame_id])

    def generate_video(self, visualization_results):
        self.vprocessor.frames2videos(visualization_results, self.output_folder)