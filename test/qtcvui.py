# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ui/pyqt5_opencv.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap,QPainter,QColor,QFont,QBrush
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QInputDialog
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1184, 719)
        MainWindow.setMouseTracking(False)
        MainWindow.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.buttonLoad = QtWidgets.QPushButton(self.centralwidget)
        self.buttonLoad.setGeometry(QtCore.QRect(150, 680, 100, 27))
        self.buttonLoad.setObjectName("buttonLoad")
        self.buttonStart = QtWidgets.QPushButton(self.centralwidget)
        self.buttonStart.setGeometry(QtCore.QRect(300, 680, 100, 27))
        self.buttonStart.setObjectName("buttonStart")
        self.buttonPause = QtWidgets.QPushButton(self.centralwidget)
        self.buttonPause.setGeometry(QtCore.QRect(450, 680, 100, 27))
        self.buttonPause.setObjectName("buttonPause")
        self.videoWidget = QtWidgets.QLabel(self.centralwidget)
        self.videoWidget.setGeometry(QtCore.QRect(0, 0, 1184, 666))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.videoWidget.setFont(font)
        self.videoWidget.setMouseTracking(True)
        self.videoWidget.setAutoFillBackground(True)
        self.videoWidget.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.videoWidget.setObjectName("videoWidget")

        self.videoTabel = QtWidgets.QTableWidget(self.centralwidget)
        self.videoTabel.setStyleSheet('background: red; border: none; color: red;')
        self.videoTabel.setShowGrid(False)

        self.videoTabel.setGeometry(QtCore.QRect(1184-600, 0, 600, 150))
        self.videoTabel.setEditTriggers(self.videoTabel.NoEditTriggers)
        self.videoTabel.setRowCount(3)
        self.videoTabel.setColumnCount(5)
        for i in range(5):
            if i == 0:
                self.videoTabel.setColumnWidth(i,100)
            else:
                self.videoTabel.setColumnWidth(i,50)
        for i in range(3):
            self.videoTabel.setRowHeight(i,50)
        self.videoTabel.setHorizontalHeaderLabels(['target','x1','y1','x2','y2'])
        for index in range(self.videoTabel.columnCount()):
            # headItem = self.videoTabel.horizontalHeaderItem(index)
            # headItem.setFont(QFont("simsun", 24, QFont.Bold))
            # headItem.setForeground(QBrush(Qt.white))
            # headItem.setBackground(QBrush(Qt.blue))
            # headItem.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            newItem = QtWidgets.QTableWidgetItem("张三")
            newItem.setForeground(QBrush(QColor(255, 0, 0)))
            newItem.setBackground(QBrush(QColor(0,255,0)))
            newItem.setTextAlignment(Qt.AlignCenter)
            newItem.setFont(QFont("simsun", 12, QFont.Bold))
            self.videoTabel.setItem(1, 0, newItem)
        self.videoTabel.setSpan(0,0,1,5)
        self.videoTabel.setHidden(True)
        self.videoTabel.verticalHeader().setVisible(False)
        self.videoTabel.horizontalHeader().setVisible(False)
        # self.videoTabel.clear()
        # self.videoTabel.resize(300,100)

        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Aerial video Demo"))
        self.buttonLoad.setText(_translate("MainWindow", "加载视频"))
        self.buttonStart.setText(_translate("MainWindow", "开始"))
        self.buttonPause.setText(_translate("MainWindow", "暂停"))
        self.videoWidget.setText(_translate("MainWindow", "<br><br><br><br><br><br><br><br><br><br><br>请选择视频文件"))

        # painter = self.videoWidget.
        # painter = self.videoWidget.initPainter(p)
        # painter.setPen(QtGui.QColor(166,66,250))
        # painter.drawRect(QtCore.QRect(400, 400, 200, 200))
        # painter.drawRect(QtCore.QRect(400, 400, 200, 200))
        # painter.end()
        # self.videoWidget.repaint()

# class ImageWindow(QtWidgets.QLabel):
#     # 绘制事件
#     def paintEvent(self, event):
#         super().paintEvent(event)
#         self.color_list = [[87, 141, 20], [5, 58, 156], [149, 64, 13]]
#         self.classes = ['起重机', '挖掘机', '打桩机']
#         rect = QtCore.QRect(400, 400, 200, 200)
#         painter = QtGui.QPainter(self)
#         painter.setPen(QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
#         painter.drawRect(rect)
#         self.draw_bottom_area()
#
#     def draw_bottom_area(self):
#         longtitude = np.random.uniform(-1, 1)
#         latitude = np.random.uniform(-0.5, 0.5)
#         bottom_str = '经度:{:.3f} 纬度:{:.3f}'.format(113.5 + longtitude, 22.5 + latitude)
#         w,h = self.geometry().width(),self.geometry().height()
#
#         rect = QtCore.QRect(0, h - h//20, w, h)
#         painter = QtGui.QPainter(self)
#         painter.setPen(QtGui.QPen(QtCore.Qt.green, 2, QtCore.Qt.SolidLine))
#         painter.drawRect(rect)
