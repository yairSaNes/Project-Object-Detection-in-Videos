import os
import numpy as np
import matplotlib.pyplot as plt

from demo.inference import run_inference

from mega_core.config import cfg


def run_inference_vid_params(vid,i, t):

    os.chdir('/home/adiyair/mega.pytorch')  # mega root

    config_base_file = 'configs/BASE_RCNN_1gpu.yaml'
    # method = 'base'
    method = 'fgfa'
    config_file = 'configs/FGFA/vid_R_50_C4_FGFA_1x.yaml'  # config file
    checkpoint_file = 'checkpoint_out/training_out_31_folders/model_final.pth'

    video_name = vid
    video_id = '0'+str(i)
    input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/'+video_name
    mode = "batch" # "single"
    output_sfx = '_odeval'
    # input_images_folder = 'datasets/ILSVRC2015_examples/ILSVRC2015_sample1/DATA/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00071005/'
    confidence_threshold = t
    output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/VID/train+ODEVAL/batchVal/'+video_id+'-'+video_name+'/'+video_id+'-'+video_name+'.thresh-'+str(confidence_threshold)
    # output_folder = 'datasets/tests'
    file_suffix = '.jpg'
    # max_num_frames = -1
    max_num_frames = -1

    output_sfx = '{}_confidence_{}'.format(output_sfx, confidence_threshold)

    run_inference(config_base_file,
                  method,
                  config_file,
                  checkpoint_file,
                  input_images_folder,
                  output_sfx,
                  confidence_threshold,
                  output_folder=output_folder,
                  file_suffix=file_suffix,
                  max_num_frames=max_num_frames,
                  )

    pass

def run_inference_vid():

    os.chdir('/home/adiyair/mega.pytorch')  # mega root

    config_base_file = 'configs/BASE_RCNN_1gpu.yaml'
    # method = 'base'
    method = 'fgfa'
    config_file = 'configs/FGFA/vid_R_50_C4_FGFA_1x.yaml'  # config file
    checkpoint_file = 'checkpoint_out/training_out_31_folders/model_final.pth'
    #checkpoint_file = 'checkpoint_out/training_out_all/model_final.pth'
    checkpoint_file = 'training_out_all_7frames/model_final.pth'
    # checkpoint_file = 'checkpoint_out/training_out_all_1frame/model_final.pth'
    # checkpoint_file = 'adi_output_try/7f/model_0045000.pth'



    # input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000268_05773_v' #highway_vid
    input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000086_00000_v'
    # input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000117_02622_v'
    # input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000137_00458_v'
    # input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000339_00001_v'
    input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000305_00000_v'
    input_images_folder = 'datasets/ILSVRC2015_VIS/Data/VID/val/uav0000182_00000_v'
    output_sfx = '_odeval'
    # input_images_folder = 'datasets/ILSVRC2015_examples/ILSVRC2015_sample1/DATA/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00071005/'
    confidence_threshold = 0.1
    confidence_threshold_rpn = 0.2

    output_folder = 'datasets/ILSVRC2015_VIS/ODEval/presentation/7'
    # output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/adi_fixed_try/8'
    # output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/RPN_TEST/8'

    # checkpoint_file = 'trained_and_fixed_anno/all_data_19frames/model_final.pth'
    # checkpoint_file = 'trained_and_fixed_anno/all_data_1frames/model_final.pth'
    checkpoint_file = 'trained_and_fixed_anno/all_data_7frames/model_final.pth'
    checkpoint_file = 'adi_output_try/7f/model_final.pth'


    # output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/VID/train+ODEVAL/classUnion/network.19.frames/val.3.frames/uav0000339_00001_v'
    # output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/VID/train+ODEVAL/classUnion/network.1.frames/val.3.frames/uav0000339_00001_v'
    # output_folder = 'datasets/ILSVRC2015_VIS/ODEval/output/VID/train+ODEVAL/classUnion/network.7.frames/val.7.frames/uav0000339_00001_v'
    # output_folder = 'datasets/tests'


    file_suffix = '.jpg'
    max_num_frames = -1

    output_sfx = '{}_confidence_{}'.format(output_sfx, confidence_threshold)

    run_inference(config_base_file,
                  method,
                  config_file,
                  checkpoint_file,
                  input_images_folder,
                  output_sfx,
                  confidence_threshold,
                  confidence_threshold_rpn,
                  output_folder=output_folder,
                  file_suffix=file_suffix,
                  max_num_frames=max_num_frames,
                  )

    pass



if __name__ == '__main__':

    run_inference_vid()

    # # threshs = np.linspace(0.1,0.9,num = 9)
    # threshs =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # # threshs = [0.6,0.7,0.8,0.9]
    # videos = [ "uav0000305_00000_v", "uav0000339_00001_v"]
    # # videos = ["uav0000086_00000_v", "uav0000117_02622_v", "uav0000137_00458_v", "uav0000182_00000_v",
    # #           "uav0000268_05773_v", "uav0000305_00000_v", "uav0000339_00001_v"]
    # for i in range(len(videos)):
    #     print(videos[i])
    #     for t in threshs:
    #         print('	   threshold = ' + str(t))
    #         run_inference_vid_params(videos[i],i+5, t)
    print('Done!')

