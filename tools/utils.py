from mega_core.data.datasets.vid import VIDDataset
CATEGORIES = VIDDataset.classes
IS_MERGE_CLASSES = True
# IS_MERGE_CLASSES = False
from analyze.bounding_box  import Box

def convert_BoxList_to_Box(box_list):

    bbox = box_list.bbox.numpy()
    bbox_type = 'ltrb'
    image_shape = box_list.size[::-1]

    extra_fields = {}
    for key, val in box_list.extra_fields.items():
        extra_fields[key] = val.numpy()



    if 'labels' in extra_fields:
        labels = extra_fields['labels']
        if IS_MERGE_CLASSES:
            '''
            class2 - 'people': 'pedestrian', 'people'
            class3 - 'bicycle': 'bicycle','tricycle', 'awning-tricycle', 'motor'
            class4 - 'car':  'car', 'van', 'truck','bus'
            '''
            merged_classes = [0,
                              2, 2, 3, 4,
                              4, 4, 3, 3,
                              4, 3, 11]
            for i in range(len(labels)):
                labels[i] = merged_classes[labels[i]]

        label_names = [CATEGORIES[i] for i in labels]
        extra_fields['labels'] = label_names


    if 'objectness' in extra_fields:
        objectness = extra_fields['objectness']
        extra_fields['scores'] = objectness
        labels = ['object' for i in objectness]
        extra_fields['labels'] = labels

    box = Box(bbox, image_shape, bbox_type=bbox_type, extra_fields=extra_fields)

    return box


def convert_vid_annotations_to_Box(ann_vid):

    bbox = ann_vid['boxes']
    image_shape = ann_vid['im_info']
    labels = ann_vid['labels']

    if IS_MERGE_CLASSES:
        # merged_classes = ['ignored-regions', # always index 0
        #     'people','people','bicycle','car',
        #     'car','car','bicycle','bicycle',
        #     'car','bicycle','others']
        # for i in range(len(labels)):
        #     labels[i] = merged_classes[labels[i]]
        '''
        class2 - 'people': 'pedestrian', 'people'
        class3 - 'bicycle': 'bicycle','tricycle', 'awning-tricycle', 'motor'
        class4 - 'car':  'car', 'van', 'truck','bus'
        '''
        merged_labels = []
        for l in labels:
            new_label = l.replace("pedestrian", "people")
            new_label = new_label.replace("awning-tricycle", "bicycle")
            new_label = new_label.replace("motor", "bicycle")
            new_label = new_label.replace("tricycle", "bicycle")
            new_label = new_label.replace("van", "car")
            new_label = new_label.replace("truck", "car")
            new_label = new_label.replace("bus", "car")
            merged_labels.append(new_label)
        labels = merged_labels

    # labels_rpn = ['1' for n in len(labels)]
    labels_rpn = ['object']*len(labels)

    extra_fields = {'labels': labels}
    extra_fields_rpn = {'labels': labels_rpn}


    box = Box(bbox, image_shape, bbox_type='ltrb', extra_fields=extra_fields)
    box_rpn = Box(bbox, image_shape, bbox_type='ltrb', extra_fields=extra_fields_rpn)

    return box, box_rpn

