"""
Faster R-CNN in Keras
---
# Reference:

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json
import random
import pickle
import pprint
import numpy as np
from optparse import OptionParser

from keras.layers import Input, Add, Dense
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D
from keras.layers import TimeDistributed
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import generic_utils
from keras.utils import plot_model
import tensorflow as tf

from .roi_pooling import RoiPoolingConv
from . import roi_helpers 
from .utils import data_generators
from .utils import config
from . import losses


class FasterRCNN(object):
    """
    Faster R-CNN
    ---
    """
    def __init__(self):
        self.C = config.Config()
        self.cnn_name = self.C.conv_net
        if self.cnn_name == 'vgg16':
            from .cnn import vgg16 as convNet
        if self.cnn_name == 'vgg19':
            from .cnn import vgg19 as convNet
        elif self.cnn_name == 'resnet50':
            from .cnn import resnet50 as convNet
        else:
            raise ValueError("CNN are limited to the following types: \n \
                            1. VGG16 \n \
                            2. VGG19 \n \
                            3. ResNet50")
        self.cnn_model = convNet
    
    def get_weight_path(self):
        return "%s_weights_tf.h5" % self.cnn_name

    def load_data_train(self):
        from .utils.parser_pascal import get_data
        self.all_imgs, self.class_count, self.class_mapping = get_data(self.C.train_path)
        print("num of classes %d" % len(self.class_count))

        if 'bg' not in self.class_count:
            self.class_count['bg'] = 0
            self.class_mapping['bg'] = len(self.class_mapping)
        
        print("training images per classes")
        pprint.pprint(self.class_count)

        print("class ids with labels")
        pprint.pprint(self.class_mapping)

        with open('./data/class_count_pascal.json', 'w') as fp_count:
            json.dump(self.class_count, fp_count, indent=4)
        with open('./data/class_mapping_pascal.json', 'w') as fp_mapping:
            json.dump(self.class_mapping, fp_mapping, indent=4)
        random.shuffle(self.all_imgs)

        train_imgs = [s for s in self.all_imgs if s['imageset'] == 'train']
        val_imgs = [s for s in self.all_imgs if s['imageset'] == 'val']
        test_imgs = [s for s in self.all_imgs if s['imageset'] == 'test']

        print('num of train samples {}'.format(len(train_imgs)))
        print('num of val samples {}'.format(len(val_imgs)))
        print('num of test samples {}'.format(len(test_imgs)))

        # groundtruth anchor
        self.data_gen_train = data_generators.get_anchor_gt(train_imgs, self.class_count, 
                                        self.C, self.cnn_model.get_img_output_length, 
                                        K.image_dim_ordering(), mode='train')
        self.data_gen_val = data_generators.get_anchor_gt(val_imgs, self.class_count, 
                                        self.C, self.cnn_model.get_img_output_length, 
                                        K.image_dim_ordering(), mode='val')
        self.data_gen_test = data_generators.get_anchor_gt(test_imgs, self.class_count, 
                                        self.C, self.cnn_model.get_img_output_length, 
                                        K.image_dim_ordering(), mode='test')
    
    def load_data_test(self, path):
        self.class_mapping = json.load(open('./data/class_mapping_pascal.json', 'r'))
        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)
        
        self.class_mapping = {v:k for k, v in self.class_mapping.items()}
        print("class ids with labels")
        pprint.pprint(self.class_mapping)

        self.class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) \
                                for v in self.class_mapping}
        
        self.test_images = []
        self.test_images_bbox = []
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print("image found", img_name)
            self.test_images.append(os.path.join(path, img_name))
            self.test_images_bbox.append(os.path.join(path, img_name + '.bbox.png'))
        
        assert len(self.test_images) == len(self.test_images_bbox)

    def region_proposal_net(self, base_layers, num_anchors):
        """
        Region Proposal Network
        --
        Args:
            base_layers:
            num_anchors:
        """
        if self.cnn_name == 'vgg16':
            proposal_dim = 256
        else:
            proposal_dim = 512

        x = Convolution2D(proposal_dim, (3, 3), 
                        padding='same', 
                        activation='relu', 
                        kernel_initializer='normal', 
                        name='proposal_conv1')(base_layers)

        x_class = Convolution2D(num_anchors, (1, 1), 
                                activation='sigmoid', 
                                kernel_initializer='uniform', 
                                name='proposal_out_class')(x)
        x_regress = Convolution2D(num_anchors * 4, (1, 1), 
                                activation='linear', 
                                kernel_initializer='zero', 
                                name='proposal_out_regress')(x)

        return [x_class, x_regress, base_layers]

    def classifier(self, base_layers, classifier_layer, input_roi, num_roi, num_class, trainable=True):
        """
        Args:
            base_layers:
            classifier_layer:
            input_roi:
            num_roi:
            num_class:
            trainable:
        """
        if self.cnn_name == 'vgg16':
            pooling_regions = 7
            input_shape = (num_roi, 7, 7, 512)
        else:
            pooling_regions = 14
            input_shape = (num_roi, 14, 14, 1024)
        
        out_roi_pool = RoiPoolingConv(pooling_regions, num_roi)(
                                    [base_layers, input_roi])
        
        if self.cnn_name == 'vgg16':
            out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
            out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
            out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
        else:
            out = classifier_layer(out_roi_pool, 
                                    input_shape=input_shape,
                                    trainable=True)
            out = TimeDistributed(Flatten())(out)
        
        # classification task
        out_class = TimeDistributed(Dense(num_class,
                                        activation='softmax',
                                        kernel_initializer='zero'),
                                    name='dense_class_{}'.format(num_class))(out)
        # regression task
        out_regress = TimeDistributed(Dense(4 * (num_class - 1),
                                            activation='linear',
                                            kernel_initializer='zero'),
                                    name='dense_regress_{}'.format(num_class))(out)
        return [out_class, out_regress]

    def _write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()

    def build(self):
        input_shape_img = (None, None, 3)
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(None, 4))
        shared_layers = self.cnn_model.nn_base(img_input, trainable=True)
        num_anchors = len(self.C.anchor_scales) * len(self.C.anchor_ratios)
        
        output_region_proposal = self.region_proposal_net(shared_layers, num_anchors)
        output_classifier = self.classifier(shared_layers,
                                            self.cnn_model.classifier_layers, 
                                            roi_input, self.C.num_roi, 
                                            num_class=len(self.class_count), trainable=True)
        
        self.model_region_proposal = Model(img_input, output_region_proposal[:2])
        self.model_classifier = Model([img_input, roi_input], output_classifier)
        self.model_all = Model([img_input, roi_input], output_region_proposal[:2] + output_classifier)

        optimizer = Adam(lr=1e-5)
        self.model_region_proposal.compile(optimizer=optimizer, 
                                    loss=[losses.rpn_loss_cls(num_anchors), 
                                        losses.rpn_loss_regr(num_anchors)])
        self.model_classifier.compile(optimizer=optimizer, 
                                    loss=[losses.class_loss_cls, 
                                        losses.class_loss_regr(len(self.class_count)-1)], 
                                    metrics={'dense_class_{}'.format(len(self.class_count)): 'accuracy'})
        self.model_all.compile(optimizer='sgd', loss='mae')

        print(self.model_all.summary())
        plot_model(self.model_region_proposal, show_shapes=True, to_file='./frcnn/images/region_proposal.png')
        plot_model(self.model_classifier, show_shapes=True, to_file='./frcnn/images/classifier.png')
        plot_model(self.model_all, show_shapes=True, to_file='./frcnn/images/model_all.png')

    def train(self):
        if not os.path.isdir(self.C.log_path):
            os.mkdir(self.C.log_path)
        
        callback = TensorBoard(self.C.log_path)
        callback.set_model(self.model_all)

        epoch_length = self.C.epoch_length
        epochs = self.C.epochs
        iter_num = 0
        train_step = 0

        losses = np.zeros((epoch_length, 5))
        rpn_accuracy_monitor = []
        rpn_accuracy_epoch = []
        best_loss = np.Inf
        start_time = time.time()

        for epoch in range(epochs):
            progbar = generic_utils.Progbar(epoch_length)
            print('Epoch {}/{}'.format(epoch + 1, epochs))

            while True:
                if len(rpn_accuracy_monitor) == epoch_length and self.C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_monitor)) / len(rpn_accuracy_monitor)
                    rpn_accuracy_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # data generator
                X, Y, img_data = next(self.data_gen_train)

                loss_rpn = self.model_region_proposal.train_on_batch(X, Y)
                self._write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)

                P_rpn = self.model_region_proposal.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], self.C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, self.C, self.class_mapping)

                if X2 is None:
                    rpn_accuracy_monitor.append(0)
                    rpn_accuracy_epoch.append(0)
                    continue
                
                # sampling positive/negative samples
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_monitor.append(len(pos_samples))
                rpn_accuracy_epoch.append((len(pos_samples)))

                if self.C.num_roi > 1:
                    if len(pos_samples) < self.C.num_roi//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, 
                                                                self.C.num_roi//2, 
                                                                replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, 
                                                                self.C.num_roi-len(selected_pos_samples), 
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, 
                                                                self.C.num_roi-len(selected_pos_samples), 
                                                                replace=True).tolist()
                    sel_samples = selected_pos_samples + selected_neg_samples
                
                else:
                    # in the extreme case where num_roi = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = self.model_classifier.train_on_batch([X, X2[:, sel_samples, :]], 
                                                                [Y1[:, sel_samples, :], 
                                                                Y2[:, sel_samples, :]])
                self._write_log(callback, 
                                ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], 
                                loss_class, train_step)
                train_step += 1

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]
                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, 
                            [('rpn_cls', np.mean(losses[:iter_num, 0])), 
                            ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])), 
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_epoch)) / len(rpn_accuracy_epoch)
                    rpn_accuracy_epoch = []

                    if self.C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    self._write_log(callback,
                            ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 
                            'mean_rpn_reg_loss', 'mean_detection_cls_loss', 'mean_detection_reg_loss', 
                            'mean_detection_acc', 'total_loss'],
                            [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, 
                            loss_rpn_regr, loss_class_cls, loss_class_regr, class_acc, curr_loss],
                            epoch)

                    if curr_loss < best_loss:
                        if self.C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                        best_loss = curr_loss
                        self.model_all.save_weights(self.C.model_path)
                    break
        
        print("traing completed.")
            
    def test(self, img_path):
        import cv2 

        self.load_data_test(path=img_path)
        self.C.horizontal_flips = False
        self.C.vertical_flips = False
        self.C.rotate_90 = False

        st = time.time()

        from .utils.data_generators import format_img_size
        from .utils.data_generators import format_img_channels
        from .utils.data_generators import format_img
        from .utils.data_generators import get_real_coordinates

        if self.cnn_name == 'vgg16' or self.cnn_name == 'vgg19':
            num_feature = 512
        else:
            num_feature = 1024
        
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_feature)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_roi, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = self.cnn_model.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_scales) * len(self.C.anchor_ratios)
        rpn_layers = self.region_proposal_net(shared_layers, num_anchors)
        classifier = self.classifier(feature_map_input, 
                                    self.cnn_model.classifier_layers, 
                                    roi_input, 
                                    self.C.num_roi, 
                                    num_class=len(self.class_mapping), 
                                    trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)
        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(self.C.model_path))
        model_rpn.load_weights(self.C.model_path, by_name=True)
        model_classifier.load_weights(self.C.model_path, by_name=True)

        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        for i in range(len(self.test_images)):
            img = cv2.imread(self.test_images[i])
            X, ratio = format_img(img, self.C)
            X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)

            R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0] // self.C.num_roi+1):
                ROIs = np.expand_dims(R[self.C.num_roi*jk:self.C.num_roi*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0] // self.C.num_roi:
                    # pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0], self.C.num_roi, curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):
                    if np.max(P_cls[0, ii, :]) < self.C.bbox_threshold or \
                        np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]
                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]
                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.class_regress_std[0]
                        ty /= C.class_regress_std[1]
                        tw /= C.class_regress_std[2]
                        th /= C.class_regress_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    
                    bboxes[cls_name].append([self.C.stride*x, 
                                            self.C.stride*y, 
                                            self.C.stride*(x+w), 
                                            self.C.stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_detections = []

            for key in bboxes:
                bbox = np.array(bboxes[key])
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, 
                                        np.array(probs[key]), overlap_thresh=0.5)
                
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]
                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    cv2.rectangle(img,(real_x1, real_y1), 
                                    (real_x2, real_y2), 
                                    (int(self.class_to_color[key][0]), 
                                    int(self.class_to_color[key][1]), 
                                    int(self.class_to_color[key][2])),
                                    2)

                    textLabel = '{}: {}'.format(key, int(100*new_probs[jk]))
                    all_detections.append((key, 100*new_probs[jk]))

                    (retval,baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    text_org = (real_x1, real_y1+10)

                    cv2.rectangle(img, (text_org[0], text_org[1]+baseLine), 
                                        (text_org[0]+retval[0]+10, text_org[1]-retval[1]-10), 
                                        (0, 0, 0), 2)
                    cv2.rectangle(img, (text_org[0],text_org[1]+baseLine), 
                                        (text_org[0]+retval[0]+10, text_org[1]-retval[1]-10), 
                                        (255, 255, 255), -1)
                    cv2.putText(img, textLabel, text_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            print('Elapsed time = {}'.format(time.time() - st))
            print(self.test_images[i], all_detections)
            if all_detections:
                cv2.imwrite(self.test_images_bbox[i], img)

