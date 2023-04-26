import os
import cv2
import torch
import numpy as np
import skimage.io as io


class CRPTorchDataset:
    def __init__(self, splits,):
        self.name = splits
        self.splits = splits.split(',')
        print('dataset.split',self.splits)

        if 'train' in self.splits:
            self.loaddata('train')
        if 'valid' in self.splits:
            self.loaddata('valid')
        if 'test' in self.name:
            self.loaddata('test')


    def loaddata(self,splits):

        path = "../../data/" + splits
        v_r_chose = os.listdir(path)
        v_r_chose.sort()

        self.video_clip = {}
        self.radar_image = {}
        self.label = {}
        count = 0
        num = 0
        global  video_sub_length,total_frames
        for data_type in v_r_chose:
            if data_type == 'radar':
                feature_1_path = os.path.join(path,data_type)
                name_list = os.listdir(feature_1_path)
                name_list.sort()
                for name in name_list:
                    name_path = os.path.join(feature_1_path,name)
                    calss_list = os.listdir(name_path)
                    calss_list.sort()
                    for label_name,calss in enumerate(calss_list):
                        calss_path = os.path.join(name_path, calss)
                        image = os.listdir(calss_path)
                        image.sort()
                        for i , image_name in enumerate(image):
                            self.label[count] = label_name
                            image_path = os.path.join(calss_path,image_name)
                            self.radar_image[count] = image_path
                            count+=1
                            if count%500 == 0:
                                print('processing radar %d/%d data ' % (count, (len(image)*len(calss_list)*len(name_list))))

            else:
                feature_1_path = os.path.join(path, data_type)
                name_list = os.listdir(feature_1_path)
                name_list.sort()
                for name in name_list:
                    name_path = os.path.join(feature_1_path, name)
                    class_list = os.listdir(name_path)
                    class_list.sort()
                    for class_name in class_list:
                        class_path = os.path.join(name_path,class_name)
                        video_sub = os.listdir(class_path)
                        video_sub.sort()
                        video_sub_length = len(video_sub)
                        for sub in video_sub:
                            video_sub_path = os.path.join(class_path,sub)
                            self.video_clip[num] = video_sub_path

                            num += 1
                            if num%500 == 0:
                                print('processing video %d/%d data ' % (num, (len(video_sub)*len(class_list)*len(name_list))))

        return self.video_clip,self.radar_image,self.label


    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        data_list = []
        video_sub_path = self.video_clip[index]
        image_list = os.listdir(video_sub_path)
        image_list.sort()
        clip = np.array([io.imread(os.path.join(video_sub_path,image_list[1].split('_')[0]+'_' + image_list[1].split('_')[1] + '_{:02d}.jpg'.format(k))) for k in range(1,25)])
        clip= np.array(clip, dtype='float32')
        clip = np.float32(clip.transpose(3,0,1,2))
        data_list.append(clip)
        radar_image_path = self.radar_image[index]
        img = cv2.imread(radar_image_path)
        list_every = np.array(img.transpose(2, 0, 1), dtype='float32')
        data_list.append(list_every)
        label = self.label[index]
        label = np.array(label, dtype='float32')
        label = torch.LongTensor(label)

        data_list.append(label)
        return data_list



class CRPEvaluator:
    def __init__(self, dataset: CRPTorchDataset):
        self.dataset = dataset
        print('dataset',dataset)

    def evaluate(self, predict_results: list):  #quesid2ans传进来的参数，预测的结果
        score = 0.
        for i,data in enumerate(predict_results):
            # print('data',data)
            predict = data[0]
            label = data[1]
            if predict == label:
                score += 1
        return (score / len(predict_results))

    def dump_result(self, quesid2ans: list, path):

        with open(path, 'a') as s:
            s.write('label,predict\n')
            for data in quesid2ans:
                s.write('{},{}\n'.format(data[1], data[0]))
            s.close()

