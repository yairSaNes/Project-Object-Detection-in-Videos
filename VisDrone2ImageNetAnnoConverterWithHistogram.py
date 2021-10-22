import os
import pandas as pd
import cv2
from tqdm import tqdm
from yattag import Doc, indent

"""
Convert VisDrone VID dataset to ImageNet format.

---------------------------------

VisDrone VID annotation format

https://github.com/VisDrone/VisDrone2018-VID-toolkit

 <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

        Name	                                                      Description
 ----------------------------------------------------------------------------------------------------------------------------------
    <frame_index>     The frame index of the video frame

     <target_id>      In the DETECTION result file, the identity of the target should be set to the constant -1. 
                      In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
	              relation of the bounding boxes in different frames.

     <bbox_left>      The x coordinate of the top-left corner of the predicted bounding box

     <bbox_top>	      The y coordinate of the top-left corner of the predicted object bounding box

    <bbox_width>      The width in pixels of the predicted object bounding box

    <bbox_height>     The height in pixels of the predicted object bounding box

      <score>	      The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                      an object instance.
                      The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in 
                      evaluation, while 0 indicates the bounding box will be ignored.

  <object_category>   The object category indicates the type of annotated object: 
                            ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), 
                            tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))

   <truncation>       The score in the DETECTION file should be set to the constant -1.
                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                      (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% °´ 50%)).

    <occlusion>	      The score in the DETECTION file should be set to the constant -1.
                      The score in the GROUNDTRUTH file indicates the fraction of objects being occluded 
                      (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% °´ 50%),
                       and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).

"""

if __name__ == '__main__':
    # video_name = ' '
    # video_name = 'uav0000013_00000_v'

    source_dir = f'C:/Library/School/Project/visDrone/VisDrone2019-VID-train/'
    files = os.listdir(source_dir + 'annotations/');
    hist = {}
    for video_name in files:
        if (video_name != "createXML"):
            ann_file = source_dir + 'annotations/' + video_name
            image_dir = source_dir + 'sequences/' + video_name[:-4]

            output_root = source_dir + 'imageNetStyleAnno/'
            output_dir = os.path.join(output_root, video_name[:-4])
            os.makedirs(output_dir, exist_ok=True)

            video_id = 1

            # set video and categories dictionaries
            videos = {
                'id': video_id,
                'name': video_name,
            }

            ann_df = pd.read_csv(ann_file)
            # ann_df = pd.read_csv(ann_file, dtype=float)
            ann_df.columns = ['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score',
                              'object_category', 'truncation', 'occlusion']
            columns_int = ['frame_index', 'target_id', 'object_category', 'truncation', 'occlusion']
            ann_df[columns_int] = ann_df[columns_int].astype('int')
            # print(ann_df)

            ann_id = 0
            image_name_template = '{}.jpg'
            image_name_length = 11
            anno_name_template = '{}'
            anno_name_length = 7  # controls number of leading zeros
            images = []
            annotations = []

            frame_id_list = sorted(ann_df.loc[:, 'frame_index'].unique().tolist())  # 269 frames
            categories = [
                {'id': 0, 'name': 'ignored-regions'},
                {'id': 1, 'name': 'pedestrian'},
                {'id': 2, 'name': 'people'},
                {'id': 3, 'name': 'bicycle'},
                {'id': 4, 'name': 'car'},
                {'id': 5, 'name': 'van'},
                {'id': 6, 'name': 'truck'},
                {'id': 7, 'name': 'tricycle'},
                {'id': 8, 'name': 'awning-tricycle'},
                {'id': 9, 'name': 'bus'},
                {'id': 10, 'name': 'motor'},
                {'id': 11, 'name': 'others'},
            ]

            for frame_id in tqdm(frame_id_list):

                image_name = image_name_template.format(frame_id).zfill(image_name_length)
                image_path = os.path.join(image_dir, image_name)
                image = cv2.imread(image_path)

                mask = ann_df.loc[:, 'frame_index'] == frame_id
                df = ann_df.loc[mask, :]

                image_dict = {
                    'file_name': image_name,
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'id': frame_id,  # ?
                    'video_id': video_id,
                    'frame_id': frame_id,  # ?
                }

                images.append(image_dict)

                frame_number = anno_name_template.format(frame_id).zfill(anno_name_length)

                doc, tag, text = Doc().tagtext()

                with tag('annotation'):
                    with tag('folder'):
                        text(video_name)
                    with tag('filename'):
                        text(frame_number)
                    with tag('source'):
                        with tag('database'):
                            text('visDrone')
                    with tag('size'):
                        with tag('width'):
                            text(image.shape[1])
                        with tag('height'):
                            text(image.shape[0])

                    for target in df.iterrows():
                        target_id = int(target[1][1])
                        bbox_xmin = int(target[1][2])
                        bbox_ymin = int(target[1][3])
                        bbox_xmax = bbox_xmin + int(target[1][4])
                        bbox_ymax = bbox_ymin + int(target[1][5])
                        object_category_num = int(target[1][7])
                        occlusion = int(target[1][9])
                        x_size = int(target[1][4])
                        y_size = int(target[1][5])
                        if x_size in hist:
                            hist[x_size] += 1
                        else:
                            hist[x_size] = 1
                        if y_size in hist:
                            hist[y_size] += 1
                        else:
                            hist[y_size] = 1
                        with tag('object'):
                            with tag('trackid'):
                                text(target_id)
                            with tag('name'):
                                text(categories[object_category_num]['name'])
                            with tag('bndbox'):
                                with tag('xmax'):
                                    text(bbox_xmax)
                                with tag('xmin'):
                                    text(bbox_xmin)
                                with tag('ymax'):
                                    text(bbox_ymax)
                                with tag('ymin'):
                                    text(bbox_ymin)
                            with tag('occluded'):
                                text(occlusion)
                            with tag('generated'):
                                text('0')

                result = indent(
                    doc.getvalue(),
                    indentation='\t',
                    newline='\r'
                )
                output_xml_path = os.path.join(output_root, video_name[:-4], frame_number + ".xml")

                with open(output_xml_path, "w") as f:
                    f.write(result)

    print(hist)
    print('Done!')
