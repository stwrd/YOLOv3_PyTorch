# coding='utf-8'
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

import torch
import torch.nn as nn

class tabel_renderer(object):
    def __init__(self, rows, cols, cellWidth, cellHeight, padding):
        self.rows = rows + 1
        self.cols = cols
        self.cellWidth = cellWidth
        self.cellHeight = cellHeight
        self.padding = padding
        self.header_space_padding = 2 * self.padding
        self.sku_size = 176
        self.boardWidth = (cols - 1) * cellWidth + padding * (cols + 1) + self.sku_size #底板宽
        self.boardHeight = self.rows * cellHeight + padding * self.rows + self.header_space_padding#底板高

        self.white_board = np.ones([self.boardHeight,self.boardWidth,3],dtype=np.uint8)*255
        self.ft = put_chinese_text('simsun.ttc')
        attribs = ['x1','y1','x2','y2']
        #绘制表头
        self.font_size = 32
        # text_size = self.ft.getTextSize('目标名称',self.font_size)
        cv2.rectangle(self.white_board,(self.padding,self.padding),(self.padding+self.sku_size,self.padding+self.cellHeight),(0, 0, 220),-1)
        # x_start_pos = self.padding + (self.sku_size - text_size[0])/2
        # y_start_pos = self.padding + (self.cellHeight - text_size[1]) / 2
        # self.white_board = self.ft.draw_text(self.white_board, (int(x_start_pos),int(y_start_pos)), '目标名称', self.font_size, [225, 255, 255])

        for i,attrib in enumerate(attribs):
            # text_size = self.ft.getTextSize(attrib, self.font_size)
            cv2.rectangle(self.white_board, (self.cellWidth*i + self.padding*(i+2) + self.sku_size, self.padding),
                          (self.cellWidth*(i+1) + self.padding*(i+2) + self.sku_size, self.padding + self.cellHeight), (0, 0, 220), -1)
            # x_start_pos = self.padding*(i+2) +self.cellWidth*i + self.sku_size +(self.cellWidth-text_size[0])/2
            # y_start_pos = self.padding + (self.cellHeight - text_size[1])/2
            # self.white_board = self.ft.draw_text(self.white_board, (int(x_start_pos), int(y_start_pos)), attrib, self.font_size,
            #                                      [225, 255, 255])

        for j in range(1,self.rows):
            rows_padding = j * self.padding + self.header_space_padding + j * self.cellHeight
            for i in range(self.cols):
                if i==0:
                    cv2.rectangle(self.white_board,(self.padding,rows_padding),(self.padding+self.sku_size,rows_padding+self.cellHeight),(80,100,255),-1)
                else:
                    cv2.rectangle(self.white_board, (self.padding*(i+1) + self.sku_size + self.cellWidth*(i-1), rows_padding),
                                  (self.padding*(i+1) + self.sku_size + self.cellWidth*i, rows_padding + self.cellHeight), (80,100,255), -1)
    def UpdateInfo(self,tabel_list,classes,bottom_img=None):
        if bottom_img is not None:
            self.white_board = cv2.addWeighted(bottom_img, 0.3, self.white_board, 0.7, 0)

        # 绘制表头
        text_size = self.ft.getTextSize('目标名称', self.font_size)
        x_start_pos = self.padding + (self.sku_size - text_size[0]) / 2
        y_start_pos = self.padding + (self.cellHeight - text_size[1]) / 2
        self.white_board = self.ft.draw_text(self.white_board, (int(x_start_pos), int(y_start_pos)), '目标名称',
                                             self.font_size, [225, 255, 255])
        attribs = ['x1', 'y1', 'x2', 'y2']
        for i, attrib in enumerate(attribs):
            text_size = self.ft.getTextSize(attrib, self.font_size)
            x_start_pos = self.padding * (i + 2) + self.cellWidth * i + self.sku_size + (
                        self.cellWidth - text_size[0]) / 2
            y_start_pos = self.padding + (self.cellHeight - text_size[1]) / 2
            self.white_board = self.ft.draw_text(self.white_board, (int(x_start_pos), int(y_start_pos)), attrib,
                                                 self.font_size,[225, 255, 255])
        r = 1
        for cls in classes:
            axis = tabel_list[cls]
            if len(axis) != 0:
                for x1,y1,x2,y2 in axis:
                    x1,y1,x2,y2 = str(x1),str(y1),str(x2),str(y2)
                    rows_padding = r * self.padding + self.header_space_padding + r * self.cellHeight
                    text_size = self.ft.getTextSize(cls, self.font_size)
                    x_start_pos = int(self.padding + (self.sku_size - text_size[0]) / 2)
                    y_start_pos = int(rows_padding + (self.cellHeight - text_size[1]) / 2)
                    self.white_board = self.ft.draw_text(self.white_board,(x_start_pos,y_start_pos),cls,self.font_size,[255,255,255])

                    #x1,y1,x2,y2
                    text_size = self.ft.getTextSize(x1, self.font_size)
                    x_start_pos = int(self.padding*2 + self.sku_size + (self.cellWidth - text_size[0]) / 2)
                    y_start_pos = int(rows_padding + (self.cellHeight - text_size[1]) / 2)
                    self.white_board = self.ft.draw_text(self.white_board, (x_start_pos, y_start_pos), x1, self.font_size,
                                      [255, 255, 255])

                    text_size = self.ft.getTextSize(y1, self.font_size)
                    x_start_pos = int(self.padding*3 + self.cellWidth + self.sku_size + (self.cellWidth - text_size[0]) / 2)
                    y_start_pos = int(rows_padding + (self.cellHeight - text_size[1]) / 2)
                    self.white_board = self.ft.draw_text(self.white_board, (x_start_pos, y_start_pos), y1, self.font_size,
                                      [255, 255, 255])

                    text_size = self.ft.getTextSize(x2, self.font_size)
                    x_start_pos = int(self.padding*4 + self.cellWidth*2 + self.sku_size + (self.cellWidth - text_size[0]) / 2)
                    y_start_pos = int(rows_padding + (self.cellHeight - text_size[1]) / 2)
                    self.white_board = self.ft.draw_text(self.white_board, (x_start_pos, y_start_pos), x2, self.font_size,
                                      [255, 255, 255])

                    text_size = self.ft.getTextSize(y2, self.font_size)
                    x_start_pos = int(self.padding*5 + self.cellWidth*3 + self.sku_size + (self.cellWidth - text_size[0]) / 2)
                    y_start_pos = int(rows_padding + (self.cellHeight - text_size[1]) / 2)
                    self.white_board = self.ft.draw_text(self.white_board, (x_start_pos, y_start_pos), y2, self.font_size,
                                      [255, 255, 255])
                    r = r+1
        return self.white_board

if __name__ == '__main__':
    tbr = tabel_renderer(7,5,100,50,2)
    tabel_list = {'起重机':[[32,23,56,78],[21,34,34,56]],'打桩机':[[12,32,45,67]],'挖掘机':[[12,34,5,67],[123,234,5678,43]]}
    classes = ['起重机', '挖掘机', '打桩机']
    tbr.UpdateInfo(tabel_list,classes)
    cv2.imshow('tt',tbr.white_board)
    cv2.waitKey(0)
