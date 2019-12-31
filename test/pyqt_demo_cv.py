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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ft2 import put_chinese_text
from TabelRenderer import tabel_renderer

import torch
import torch.nn as nn
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from common.utils import non_max_suppression, bbox_iou
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QWidget

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

        # connect signals
        self.buttonLoad.clicked.connect(self.load_file)
        self.buttonStart.clicked.connect(self.start_video)
        self.buttonPause.clicked.connect(self.pause_video)

        # Video saver.
        self.videoSaver = None

        #detect
        self.ft = put_chinese_text('simsun.ttc')

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
        # super(Qtcv, self).resizeEvent(QResizeEvent)
        self.resize(self.geometry().width(),self.geometry().height())
        self.buttonLoad.setGeometry(QtCore.QRect(150, self.geometry().height() - 40, 100, 27))
        self.buttonStart.setGeometry(QtCore.QRect(300, self.geometry().height() - 40, 100, 27))
        self.buttonPause.setGeometry(QtCore.QRect(450, self.geometry().height() - 40, 100, 27))
        self.videoWidget.setGeometry(QtCore.QRect(0, 0, self.geometry().width(), self.geometry().height() -60))

    def _draw_frame(self, frame):
        # convert to pixel
        self.frameSize = (self.videoWidget.geometry().width(), self.videoWidget.geometry().height())
        # frame = cv2.resize(frame, (self.frameSize[0], self.frameSize[1]))
        cvtFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cvtFrame = frame
        img = QImage(cvtFrame, cvtFrame.shape[1], cvtFrame.shape[0], QImage.Format_RGB888)#.scaled(self.frameSize[0],self.frameSize[1])
        pix = QPixmap.fromImage(img).scaled(self.frameSize[0],self.frameSize[1])
        # pix = QPixmap.fromImage(img)
        self.videoWidget.setPixmap(pix)

    def _next_frame(self):
        try:
            if self.capture is not None:
                _ret, frame = self.capture.read()
                if frame is None:
                    print("ERROR: Read next frame failed with returned value {}.".format(_ret))

                # detect
                if self.idx % 6 ==  0:
                    # start_t = cv2.getTickCount()
                    frame = self.detect(frame)
                    # end_t = cv2.getTickCount()
                    # print('total cost time ',(end_t-start_t)*1000/cv2.getTickFrequency())

                    # Draw.
                    self._draw_frame(frame)
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
        resize_img = cv2.resize(img_4k,(1280,768))
        for x in range(0, 960, 320):
            for y in range(0, 576, 192):
                roi = resize_img[y:y + 384, x:x + 640, :]
                offset = (x, y)
                images.append({'image': roi, 'offset': offset})
        return images

    def resize_square(self, img, width=640, height=320, color=(0, 0, 0)):  # resize a rectangular image to a padded square
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)  # resized, no border
        return img

    def plot_one_box(self, x, img, color, label=None, line_thickness=None):  # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
        color = color
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            # 绘制标签
            text_size = tl * 4
            t_size = self.ft.getTextSize(label, text_size)

            text_lt = c1[0], c1[1] - t_size[1]
            text_rb = c1[0] + t_size[0], c1[1]

            cv2.rectangle(img, text_lt, text_rb, color, thickness=tl)  # filled
            cv2.rectangle(img, text_lt, text_rb, color, -1)  # filled

            img = self.ft.draw_text(img, (text_lt[0], text_lt[1]), label, text_size, [225, 255, 255])
            return img

    def draw_bottom_area(self,img, tl):
        # bottom_img = img.copy()
        longtitude = np.random.uniform(-1, 1)
        latitude = np.random.uniform(-0.5, 0.5)
        bottom_str = '经度:{:.3f} 纬度:{:.3f}'.format(113.5 + longtitude, 22.5 + latitude)
        # bottom_size = cv2.getTextSize(bottom_str, 0, fontScale=tl / 3, thickness=tf)[0]
        text_size = tl * 4
        bottom_size = self.ft.getTextSize(bottom_str, text_size=text_size)
        pad = max(int(bottom_size[1] / 2), 1)  # 添加行距
        cv2.rectangle(img, (0, img.shape[0] - bottom_size[1] - pad * 2),
                      (img.shape[1], img.shape[0]), (127, 127, 127), -1)  # filled
        # img = cv2.addWeighted(bottom_img, 0.5, img, 0.5, 0)
        # cv2.putText(img, bottom_str, (0, img.shape[0] - pad), 0, tl / 3, [225, 255, 255], thickness=tf,
        #             lineType=cv2.LINE_AA)
        img = self.ft.draw_text(img, (0, img.shape[0] - bottom_size[1] - pad), bottom_str, text_size, [225, 255, 255])
        return img

    def draw_warning_area(self, img, num, font_size=18, board_size = (300, 50)):
        font_size = font_size
        head_str = '预警: {} 个非法目标'.format(num)
        text_size = self.ft.getTextSize(head_str, font_size)

        cv2.rectangle(img, (img.shape[1] - board_size[0], 0), (img.shape[1], board_size[1]), (0, 0, 220), -1)  # filled
        x_start_pos = img.shape[1] - board_size[0] + (board_size[0] - text_size[0]) / 2
        y_start_pos = (board_size[1] - text_size[1]) / 2
        img = self.ft.draw_text(img, (int(x_start_pos), int(y_start_pos)), head_str, font_size, [225, 255, 255])
        return img

    def detect(self, frame):
        # start_t = cv2.getTickCount()
        offsetes = []
        image_origin = frame
        sub_images = self.split4K(image_origin, (1920, 1080))
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('split4K cost time ', cost_time)

        images = []
        total_outputs = None
        # start_t = cv2.getTickCount()
        images = np.empty([9,384,640,3],dtype=np.uint8)
        for i,sub_image in enumerate(sub_images):
            images[i,...] = sub_image['image']
            # img = sub_image['image']
            # img = img[:, :, ::-1].transpose(2, 0, 1)
            # img = np.ascontiguousarray(img, dtype=np.float32)
            # img /= 255.0
            # images.append(img)
            offsetes.append(sub_image['offset'])
        images = images[:,:,:,::-1].transpose(0,3,1,2)
        images = np.ascontiguousarray(images, dtype=np.float32)
        images /= 255.0
        # end_t = cv2.getTickCount()
        # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        # print('pre process cost time ', cost_time)

        # start_t = cv2.getTickCount()
        # images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        with torch.no_grad():
            outputs = self.net(images)
            # end_t = cv2.getTickCount()
            # cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
            # print('inference cost time ', cost_time)
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


        # write result images. Draw bounding boxes and labels of detections
        # classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
        classes = ['起重机', '挖掘机', '打桩机']
        img = cv2.resize(image_origin,(3840,2160))
        sub_image_shape = (2160, 3840)
        tl = round(0.002 * max(img.shape[0:2])) + 1

        # 绘制目标
        start_t = cv2.getTickCount()
        # 绘制底边
        img = self.draw_bottom_area(img, tl)

        for idx, boxes in enumerate(batch_detections):
            # The amount of padding that was added
            pad_x = 0
            pad_y = 0
            # Image height and width after padding is removed
            unpad_h = self.config["img_h"] - pad_y
            unpad_w = self.config["img_w"] - pad_x

            # Draw bounding boxes and labels of detections
            if boxes is not None:
                color_list = [[87, 141, 20], [5, 58, 156], [149, 64, 13]]
                bbox_colors = color_list

                target_list = {classes[0]: [], classes[1]: [], classes[2]: []}
                for i, box in enumerate(boxes):
                    # Rescale coordinates to original dimensions
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    box_h = ((y2 - y1) / unpad_h) * sub_image_shape[0]
                    box_w = ((x2 - x1) / unpad_w) * sub_image_shape[1]
                    y1 = (((y1 - pad_y // 2) / unpad_h) * sub_image_shape[0]).round().item()
                    x1 = (((x1 - pad_x // 2) / unpad_w) * sub_image_shape[1]).round().item()
                    x2 = (x1 + box_w).round().item()
                    y2 = (y1 + box_h).round().item()
                    x1, y1, x2, y2 = int(max(x1/2, 0)), int(max(y1/2, 0)), int(max(x2/2, 0)), int(max(y2/2, 0))

                    # Add the bbox to the plot
                    label = '%s %.2f' % (classes[int(box[-1])], box[-3])
                    color = bbox_colors[int(box[-1])]
                    # img = resize_square(img, width=640, height=320)
                    target_list[classes[int(box[-1])]].append([x1, y1, x2, y2])
                    if int(box[-1]) == 0 or True:
                        # bottom_img = img.copy()
                        # cv2.rectangle(bottom_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        # img = cv2.addWeighted(bottom_img, 0.2, img, 0.8, 0)
                        img = self.plot_one_box([x1, y1, x2, y2], img, label=label, color=color)

                img = self.draw_warning_area(img, len(boxes),font_size=18, board_size = (600, 50))
                tbr = tabel_renderer(len(boxes), 5, 80, 30, 2)
                x_padding, y_padding = 600-tbr.boardWidth, 50
                xmin, xmax, ymin, ymax = img.shape[1] - tbr.boardWidth - x_padding, img.shape[1] - x_padding, y_padding, tbr.boardHeight + y_padding
                # src_roi_img = img[ymin:ymax, xmin:xmax, :]
                tabel_img = tbr.UpdateInfo(target_list, classes)
                img[ymin:ymax, xmin:xmax, :] = tabel_img

        # add logo
        x_padding, y_padding = 50, 50
        xmin, xmax, ymin, ymax = x_padding, self.logo.shape[1] + x_padding, y_padding, self.logo.shape[0] + y_padding
        src_roi_img = img[ymin:ymax, xmin:xmax, :]
        src_roi_img[self.logo_mask] = self.logo[self.logo_mask]
        end_t = cv2.getTickCount()
        cost_time = (end_t - start_t) * 1000 / cv2.getTickFrequency()
        print('draw cost time ', cost_time)
        return img


if __name__ == '__main__':
    Qtcv.run()