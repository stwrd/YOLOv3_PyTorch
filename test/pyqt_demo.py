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

import torch
import torch.nn as nn
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap,QPainter,QColor,QFont,QBrush
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QInputDialog

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
        self.detect_size = (1280, 768)#4k图像的总缩放尺寸

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
        # Load and initialize network
        self.net = ModelMain(self.config, is_training=False)
        self.net.train(False)
        self.net.load_darknet_weights('../weights/remote_sensing.weights')
        self.net = self.net.cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        self.frame = None
        self.fps = 5

        # regist logo
        self.logo = cv2.imread('logo.png')
        self.logo = cv2.resize(self.logo, None,fx=1,fy=1, interpolation=cv2.INTER_AREA)
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
        # print('图片复制耗时 ',(cv2.getTickCount()-start_t)*1000/cv2.getTickFrequency())

        # add logo
        x_padding, y_padding = 50, 50
        xmin, xmax, ymin, ymax = x_padding, self.logo.shape[1] + x_padding, y_padding, self.logo.shape[0] + y_padding
        src_roi_img = img[ymin:ymax, xmin:xmax, :]
        src_roi_img[self.logo_mask] = self.logo[self.logo_mask]
        # add_logo_cost_time = (cv2.getTickCount()-start_t)*1000/cv2.getTickFrequency()
        # print('add_logo_cost_time cost time ',add_logo_cost_time)

        # convert to pixel
        self.frameSize = (self.videoWidget.geometry().width(), self.videoWidget.geometry().height())
        # s1 = cv2.getTickCount()
        # cvtFrame = frame
        # new_img = cv2.resize(img, self.frameSize)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(new_img, new_img.shape[1], new_img.shape[0], QImage.Format_RGB888).scaled(self.frameSize[0],self.frameSize[1],Qt.IgnoreAspectRatio,Qt.FastTransformation)#.rgbSwapped()
        pix = QPixmap.fromImage(qimg)#.scaled(self.frameSize[0],self.frameSize[1])
        # cvt_cost_time = (cv2.getTickCount()-s1)*1000/cv2.getTickFrequency()
        # print('转换耗时 ',cvt_cost_time)

        # pix = QPixmap.fromImage(img)


        #绘制底部区域
        painter = QPainter(pix)
        painter.setBrush(QColor(128,128,128))
        painter.setPen(QColor(128,128,128))
        rect = QtCore.QRect(0, self.frameSize[1] - self.frameSize[1] // 40, self.frameSize[0], self.frameSize[1] // 40)
        painter.drawRect(rect)

        longtitude = np.random.uniform(-1, 1)
        latitude = np.random.uniform(-0.5, 0.5)
        bottom_str = 'longitude:{:.3f} latitude:{:.3f}'.format(113.5 + longtitude, 22.5 + latitude)
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
                # s1 = cv2.getTickCount()
                color_list = [[87, 141, 20], [5, 58, 156], [149, 64, 13]]
                classes = ['crane', 'excavator', 'piledriver']
                bbox_colors = color_list

                t_row = len(boxes)
                self.videoTabel.setGeometry(QtCore.QRect(self.frameSize[0] - 390, 0, 390, fontsize*3 + fontsize*t_row*2))
                self.videoTabel.setRowCount(t_row+1)
                self.videoTabel.setColumnCount(5)
                for i in range(5):
                    if i == 0:
                        self.videoTabel.setColumnWidth(i, 150)
                    else:
                        self.videoTabel.setColumnWidth(i, 60)

                warning_str = 'Warning:{} Illegal target detected'.format(t_row)
                self.videoTabel.setSpan(0,0,1,5)
                newItem = QtWidgets.QTableWidgetItem(warning_str)
                newItem.setForeground(QBrush(QColor(255, 255, 255)))
                newItem.setTextAlignment(Qt.AlignCenter)
                newItem.setFont(QFont("simsun", fontsize, QFont.Bold))
                self.videoTabel.setRowHeight(0, fontsize * 3)
                self.videoTabel.setItem(0, 0, newItem)

                target_list = {classes[0]: [], classes[1]: [], classes[2]: []}
                for i, box in enumerate(boxes):
                    # Rescale coordinates to original dimensions
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    box_h = (((y2 - y1) / unpad_h) * self.frameSize[1]).round().item()
                    box_w = (((x2 - x1) / unpad_w) * self.frameSize[0]).round().item()
                    y1 = (((y1) / unpad_h) * self.frameSize[1]).round().item()
                    x1 = (((x1) / unpad_w) * self.frameSize[0]).round().item()
                    x1 = int(max(x1, 0))
                    y1 = int(max(y1, 0))
                    rect = QtCore.QRect(x1, y1, min(self.frameSize[0]-x1,box_w),min(box_h,self.frameSize[1]-y1))

                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(box[-1])], box[-3])
                    target_list[classes[int(box[-1])]].append([x1, y1, x2, y2])
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

                    newItem = QtWidgets.QTableWidgetItem(classes[int(box[-1])])
                    newItem.setForeground(QBrush(QColor(255, 255, 255)))
                    newItem.setTextAlignment(Qt.AlignCenter)
                    newItem.setFont(QFont("simsun", fontsize*2//3, QFont.Bold))
                    self.videoTabel.setRowHeight(i+1, fontsize*2)
                    self.videoTabel.setItem(i+1, 0, newItem)
                    box = [x1,y1,int(x1+box_w),int(y1+box_h)]
                    for j in range(4):
                        newItem = QtWidgets.QTableWidgetItem(str(box[j]))
                        newItem.setForeground(QBrush(QColor(255, 255, 255)))
                        newItem.setTextAlignment(Qt.AlignCenter)
                        newItem.setFont(QFont("simsun", fontsize*2//3, QFont.Bold))
                        self.videoTabel.setItem(i+1, j+1, newItem)
                self.videoTabel.setHidden(False)
                # draw_box_time = (cv2.getTickCount()-s1)*1000/cv2.getTickFrequency()
                # print('draw box cost time ', draw_box_time)
            else:
                self.videoTabel.setHidden(True)

        painter.end()
        self.videoWidget.setPixmap(pix)
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('total draw cost time ', cost_time)

    def _next_frame(self):
        try:
            if self.capture is not None:
                _ret, frame = self.capture.read()
                if frame is None:
                    print("ERROR: Read next frame failed with returned value {}.".format(_ret))
                    self.pause_video()
                    self.idx = 0
                    if self.capture is not None:
                        self.capture.release()
                    self.capture = cv2.VideoCapture(self.videoFileName)
                    self.start_video()
                else:
                    # detect
                    if self.idx % 1 ==  0:
                        # start_t = cv2.getTickCount()
                        detections = self.detect(frame)
                        # print('detections cost time ', (cv2.getTickCount() - start_t) * 1000 / cv2.getTickFrequency())

                        # Draw.
                        # draw_s = cv2.getTickCount()
                        self._draw_frame(frame,detections)
                        # end_t = cv2.getTickCount()
                        # print('_draw_frame cost time ', (end_t - draw_s) * 1000 / cv2.getTickFrequency())
                        # print('total cost time ',(end_t-start_t)*1000/cv2.getTickFrequency())
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
        # start_t = cv2.getTickCount()
        image_origin = frame
        # sub_images = self.split4K(image_origin, (2560, 1440),grid=(2,2))
        resize_img = cv2.resize(image_origin, self.detect_size)
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('split4K cost time ', cost_time)

        # start_t = cv2.getTickCount()
        images = resize_img[:,:,::-1].transpose(2,0,1)
        images = np.ascontiguousarray(images, dtype=np.float32)
        images /= 255.0
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('pre process cost time ', cost_time)

        # start_t = cv2.getTickCount()
        # images = np.asarray(images)
        images = torch.from_numpy(images).cuda().unsqueeze(0)
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('数据上传到gpu时间 cost time ', cost_time)
        with torch.no_grad():
            # s_infer = cv2.getTickCount()
            outputs = self.net(images)
            # end_t = cv2.getTickCount()
            # cost_time = (end_t - s_infer) * 1000 / cv2.getTickFrequency()
            # print('纯推理时间 cost time ', cost_time)

            total_outputs = outputs.view(-1, 8).unsqueeze(0)
            s_nms = cv2.getTickCount()
            batch_detections = non_max_suppression(total_outputs, self.config["yolo"]["classes"],
                                                   conf_thres=self.config["confidence_threshold"],
                                                   nms_thres=0.2)
        #     cost_time = (cv2.getTickCount() - s_nms) * 1000 / cv2.getTickFrequency()
        #     print('nms cost time ', cost_time)
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('每帧检测时间 ', cost_time)
        return batch_detections


if __name__ == '__main__':
    Qtcv.run()