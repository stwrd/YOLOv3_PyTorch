# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './ui/pyqt5_opencv.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

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

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "4K视频检测"))
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

class ImageWindow(QtWidgets.QLabel):
    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QtCore.QRect(400, 400, 200, 200)
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
        painter.drawRect(rect)