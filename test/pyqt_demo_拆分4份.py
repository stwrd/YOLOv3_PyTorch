# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import os
import sys
import datetime
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap,QPainter,QColor
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QInputDialog

# local import
import qtcvui

import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random
from PIL import Image, ImageFont, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from TabelRenderer import tabel_renderer

import torch
import torch.nn as nn
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou
from PyQt5 import QtCore, QtGui, QtWidgets

# Python 2/3 compatibility
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


class Qtcv(QMainWindow, qtcvui.Ui_MainWindow):
    def __init__(self):
        super(Qtcv, self).__init__()
        self.setupUi(self)

        # init parameters
        self.timer = None
        self.frame = None
        self.capture = None
        self.videoFileName = None
        self.isVideoFileLoaded = False

        # tracking parameters
        self.selection = None
        self.dragStart = None
        self.showBackproj = False
        self.trackWindow = None
        self.timestamps = []
        self.trackPoints = []
        self.movePoints = []  # movement in real world
        self.mouseOffset = (0, 0)

        # streaming parameters
        self.fps = int(30)
        self.frameSize = (self.videoWidget.geometry().width(), self.videoWidget.geometry().height())
        self.frameRatio = 1
        self.detect_size = (960, 576)#4k图像的总缩放尺寸

        # connect signals
        self.buttonLoad.clicked.connect(self.load_file)
        self.buttonStart.clicked.connect(self.start_video)
        self.buttonPause.clicked.connect(self.pause_video)

        # Video saver.
        self.videoSaver = None

        #detect
        # self.ft = put_chinese_text('simsun.ttc')
        self.font_bottom = ImageFont.truetype('simsun.ttc',24,1)
        self.font_top = ImageFont.truetype('simsun.ttc', 24,1)
        self.font_box = ImageFont.truetype('simsun.ttc', 18,1)
        self.font_tabel = ImageFont.truetype('simsun.ttc', 18,1)

        params_path = 'params.py'
        self.config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
        self.config["batch_size"] *= len(self.config["parallels"])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        # Load and initialize network
        self.net = ModelMain(self.config, is_training=False)
        self.net.train(False)
        self.net.load_darknet_weights('../weights/remote_sensing.weights')
        self.net = self.net.cuda()
        self.frame = None
        self.fps = 5

        # regist logo
        self.logo = cv2.imread('logo.png')
        self.logo = cv2.resize(self.logo, None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
        self.logo_mask = (self.logo<180)
        self.idx = 0

    def closeEvent(self, QCloseEvent):
        super(Qtcv, self).closeEvent(QCloseEvent)

    def resizeEvent(self,QResizeEvent):
        super(Qtcv, self).resizeEvent(QResizeEvent)
        self.resize(self.geometry().width(),self.geometry().height())

        self.buttonLoad.setGeometry(QtCore.QRect(150, self.geometry().height() - 40, 100, 27))
        self.buttonStart.setGeometry(QtCore.QRect(300, self.geometry().height() - 40, 100, 27))
        self.buttonPause.setGeometry(QtCore.QRect(450, self.geometry().height() - 40, 100, 27))
        self.videoWidget.setGeometry(QtCore.QRect(0, 0, self.geometry().width(), self.geometry().height() -60))
        self.frameSize = (self.videoWidget.geometry().width(), self.videoWidget.geometry().height())
        print('framesize:', self.frameSize[0], ' ', self.frameSize[1])

    def _draw_frame(self, frame,detections):
        # write result images. Draw bounding boxes and labels of detections
        # classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        start_t = cv2.getTickCount()
        sub_image_shape = (2160, 3840)#需要显示的原图尺寸
        # sub_image_shape = (self.frameSize[1],self.frameSize[0])
        if frame.shape[0] != sub_image_shape[0] or frame.shape[1] != sub_image_shape[1]:
            img = cv2.resize(frame,(sub_image_shape[1],sub_image_shape[0]))
        else:
            img = frame
        print('图片复制耗时 ',(cv2.getTickCount()-start_t)*1000/cv2.getTickFrequency())

        # add logo
        x_padding, y_padding = 50, 50
        xmin, xmax, ymin, ymax = x_padding, self.logo.shape[1] + x_padding, y_padding, self.logo.shape[0] + y_padding
        src_roi_img = img[ymin:ymax, xmin:xmax, :]
        src_roi_img[self.logo_mask] = self.logo[self.logo_mask]
        add_logo_cost_time = (cv2.getTickCount()-start_t)*1000/cv2.getTickFrequency()
        print('add_logo_cost_time cost time ',add_logo_cost_time)

        # convert to pixel
        self.frameSize = (self.videoWidget.geometry().width(), self.videoWidget.geometry().height())
        s1 = cv2.getTickCount()
        # cvtFrame = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)#.scaled(self.frameSize[0],self.frameSize[1])
        pix = QPixmap.fromImage(img).scaled(self.frameSize[0],self.frameSize[1])
        cvt_cost_time = (cv2.getTickCount()-s1)*1000/cv2.getTickFrequency()
        print('转换耗时 ',cvt_cost_time)
        # pix = QPixmap.fromImage(img)


        #绘制底部区域
        painter = QPainter(pix)
        painter.setBrush(QColor(128,128,128))
        painter.setPen(QColor(128,128,128))
        rect = QtCore.QRect(0, self.frameSize[1] - self.frameSize[1] // 40, self.frameSize[0], self.frameSize[1] // 40)
        painter.drawRect(rect)

        longtitude = np.random.uniform(-1, 1)
        latitude = np.random.uniform(-0.5, 0.5)
        bottom_str = '经度:{:.3f} 纬度:{:.3f}'.format(113.5 + longtitude, 22.5 + latitude)
        painter.setPen(QColor(255, 255, 255))
        fontsize = (self.frameSize[1] //40)-2
        painter.setFont(QtGui.QFont("simsun",fontsize, QtGui.QFont.Bold))
        painter.drawText(rect,QtCore.Qt.AlignCenter,bottom_str)

        #绘制bbox
        for idx, boxes in enumerate(detections):
            # Image height and width after padding is removed
            unpad_h = self.detect_size[1]
            unpad_w = self.detect_size[0]

            # Draw bounding boxes and labels of detections
            if boxes is not None:
                s1 = cv2.getTickCount()
                color_list = [[87, 141, 20], [5, 58, 156], [149, 64, 13]]
                classes = ['起重机', '挖掘机', '打桩机']
                bbox_colors = color_list

                target_list = {classes[0]: [], classes[1]: [], classes[2]: []}

                for i, box in enumerate(boxes):
                    # Rescale coordinates to original dimensions
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    box_h = ((y2 - y1) / unpad_h) * self.frameSize[1]
                    box_w = ((x2 - x1) / unpad_w) * self.frameSize[0]
                    y1 = (((y1) / unpad_h) * self.frameSize[1]).round().item()
                    x1 = (((x1) / unpad_w) * self.frameSize[0]).round().item()
                    x1 = int(max(x1, 0))
                    y1 = int(max(y1, 0))
                    rect = QtCore.QRect(x1, y1, min(self.frameSize[0]-x1,box_w),min(box_h,self.frameSize[1]-y1))

                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(box[-1])], box[-3])
                    label_fontsize = fontsize//2
                    font = QtGui.QFont("simsun",label_fontsize, QtGui.QFont.Bold)
                    metric = QtGui.QFontMetricsF(font)
                    font_width,font_height = metric.width(label),metric.height()
                    x1 = int(max(x1, 0))
                    y1 = int(max(y1-font_height, 0))
                    rect_bottom = QtCore.QRect(x1, y1, min(self.frameSize[0]-font_width,font_width), min(self.frameSize[1]-font_height,font_height))
                    color = bbox_colors[int(box[-1])]
                    painter.setPen(QtGui.QPen(QColor(*color), 2, QtCore.Qt.SolidLine))
                    painter.setBrush(QColor(*color))
                    painter.drawRect(rect_bottom)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawRect(rect)
                    painter.setFont(font)
                    painter.setPen(QtGui.QPen(QColor(255,255,255), 2, QtCore.Qt.SolidLine))
                    painter.drawText(rect_bottom, QtCore.Qt.AlignLeft, label)

                draw_box_time = (cv2.getTickCount()-s1)*1000/cv2.getTickFrequency()
                print('draw box cost time ', draw_box_time)

        painter.end()
        self.videoWidget.setPixmap(pix)
        end_t = cv2.getTickCount()
        cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        print('total draw cost time ', cost_time)

    def _next_frame(self):
        try:
            if self.capture is not None:
                _ret, frame = self.capture.read()
                if frame is None:
                    print("ERROR: Read next frame failed with returned value {}.".format(_ret))

                # detect
                if self.idx % 1 ==  0:
                    start_t = cv2.getTickCount()
                    detections = self.detect(frame)

                    end_t = cv2.getTickCount()
                    print('total cost time ',(end_t-start_t)*1000/cv2.getTickFrequency())
                    # frame = cv2.resize(frame, (self.frameSize[0], self.frameSize[1]))
                    # print('resize framesize:',self.frameSize[0],' ', self.frameSize[1])

                    # Draw.
                    self._draw_frame(frame,detections)
                self.idx+=1

        except Exception as e:
            # TODO: Distinguish Exception types.
            # Pause video.
            self.pause_video()
            print("Error: Exception while reading next frame.")
            print(str(e))

    def load_file(self):
        try:
            self.pause_video()
            self.videoFileName = QFileDialog.getOpenFileName(self, 'Select .h264 Video File')[0]
            self.isVideoFileLoaded = True

            if self.capture is not None:
                self.capture.release()
            self.capture = cv2.VideoCapture(self.videoFileName)

            # get frame ratio to shrink
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.frameRatio = min(self.frameSize[0] / width, self.frameSize[1] / height)

            # get the first frame
            self._next_frame()

        except Exception as e:
            self.capture = None
            self.isVideoFileLoaded = False
            print("Error: Exception while selecting&opening video file")
            print(str(e))

        finally:
            self.timestamps = []
            self.trackPoints = []

    def start_video(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self.timer.start(1)

    def pause_video(self):
        try:
            self.timer.stop()
            print("INFO: Streaming paused.")
        except Exception as e:
            print("Error: Exception while pausing")
            print(str(e))

    @staticmethod
    def run():
        # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # Enable scaling.
        app = QApplication(sys.argv)  # A new instance of QApplication
        form = Qtcv()  # We set the form to be our ExampleApp (design)
        form.show()  # Show the form
        app.exec_()  # And execute the app

    def split4K(self, img_4k, size, grid=(3, 3)):
        images = []
        resize_img = cv2.resize(img_4k,self.detect_size)
        for x in range(0, 640, 320):
            for y in range(0, 384, 192):
                roi = resize_img[y:y + 384, x:x + 640, :]
                offset = (x, y)
                images.append({'image': roi, 'offset': offset})
        return images

    def resize_square(self, img, width=640, height=320, color=(0, 0, 0)):  # resize a rectangular image to a padded square
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)  # resized, no border
        return img

    def plot_one_box(self, x, img, draw, color, label=None, line_thickness=None):  # Plots one bounding box on image img
        color = color
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        draw.rectangle((c1[0],c1[1],c2[0],c2[1]),outline=tuple(color),width=5)
        if label:
            # 绘制标签
            t_size = self.font_box.getsize(label)
            text_lt = c1[0], c1[1] - t_size[1]
            text_rb = c1[0] + t_size[0], c1[1]
            draw.rectangle((text_lt[0],text_lt[1],text_rb[0],text_rb[1]),fill=tuple(color),width=5)
            draw.text((text_lt[0],text_lt[1]),label,fill='white',font=self.font_box)

    def draw_bottom_area(self,img,draw):
        longtitude = np.random.uniform(-1, 1)
        latitude = np.random.uniform(-0.5, 0.5)
        bottom_str = '经度:{:.3f} 纬度:{:.3f}'.format(113.5 + longtitude, 22.5 + latitude)

        bottom_size = self.font_bottom.getsize(bottom_str)
        pad = max(int(bottom_size[1] / 2), 1)  # 添加行距
        draw.rectangle((0, img.height - bottom_size[1] - pad * 2,img.width, img.height),fill='gray')
        draw.text((0,img.height - pad - bottom_size[1]),bottom_str,'white',self.font_bottom)

    def draw_warning_area(self, img,draw, num, font_size=18):
        head_str = '预警: {} 个非法目标'.format(num)
        text_size = self.font_top.getsize(head_str)
        board_size = (300, 50)
        draw.rectangle((img.width - board_size[0], 0, img.width, board_size[1]), fill=(220, 0, 0))
        x_start_pos = img.width - board_size[0] + (board_size[0] - text_size[0]) / 2
        y_start_pos = (board_size[1] - text_size[1]) / 2
        draw.text((x_start_pos,y_start_pos),head_str,fill='white',font=self.font_top)

    def detect(self, frame):
        start_t = cv2.getTickCount()
        offsetes = []
        image_origin = frame
        sub_images = self.split4K(image_origin, (2560, 1440),grid=(2,2))
        end_t = cv2.getTickCount()
        cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        print('split4K cost time ', cost_time)

        images = []
        total_outputs = None
        start_t = cv2.getTickCount()
        images = np.empty([4,384,640,3],dtype=np.uint8)
        for i,sub_image in enumerate(sub_images):
            images[i,...] = sub_image['image']
            offsetes.append(sub_image['offset'])
        images = images[:,:,:,::-1].transpose(0,3,1,2)
        images = np.ascontiguousarray(images, dtype=np.float32)
        images /= 255.0
        end_t = cv2.getTickCount()
        cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        print('pre process cost time ', cost_time)

        start_t = cv2.getTickCount()
        # images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        with torch.no_grad():
            outputs = self.net(images)
            end_t = cv2.getTickCount()
            cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
            print('纯推理时间 cost time ', cost_time)
            for i, offset in enumerate(offsetes):
                outputs[i, :, 0] += offset[0]
                outputs[i, :, 1] += offset[1]

            total_outputs = outputs.view(-1, 8).unsqueeze(0)
            batch_detections = non_max_suppression(total_outputs, self.config["yolo"]["classes"],
                                                   conf_thres=self.config["confidence_threshold"],
                                                   nms_thres=0.2)
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('inference cost time ', cost_time)
        return batch_detections


if __name__ == '__main__':
    Qtcv.run()