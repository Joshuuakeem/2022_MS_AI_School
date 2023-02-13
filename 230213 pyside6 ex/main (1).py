# -*- coding: utf-8 -*-

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)
import sys


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        if not self.objectName():
            self.setObjectName(u"MainWindow")
        self.resize(1280, 800)
        self.setMinimumSize(QSize(1280, 800))
        self.setMaximumSize(QSize(1280, 800))
        self.setSizeIncrement(QSize(1280, 800))
        self.setBaseSize(QSize(1280, 800))
        self.open = QAction(self)
        self.open.setObjectName(u"open")
        self.save = QAction(self)
        self.save.setObjectName(u"save")
        self.close = QAction(self)
        self.close.setObjectName(u"close")
        self.action = QAction(self)
        self.action.setObjectName(u"action")
        self.action_2 = QAction(self)
        self.action_2.setObjectName(u"action_2")
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.paint = QPushButton(self.centralwidget)
        self.paint.setObjectName(u"paint")
        self.paint.setMinimumSize(QSize(0, 50))
        icon = QIcon()
        icon.addFile(u":/ann/paint.png", QSize(), QIcon.Normal, QIcon.Off)
        self.paint.setIcon(icon)
        self.paint.setIconSize(QSize(25, 25))

        self.verticalLayout_2.addWidget(self.paint)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.img_label = QLabel(self.centralwidget)
        self.img_label.setObjectName(u"img_label")

        self.horizontalLayout.addWidget(self.img_label)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 23)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1280, 30))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.open)
        self.menu.addAction(self.save)
        self.menu.addAction(self.close)

        self.retranslateUi()

        QMetaObject.connectSlotsByName(self)
    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.open.setText(QCoreApplication.translate("MainWindow", u"\uc5f4\uae30", None))
#if QT_CONFIG(shortcut)
        self.open.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.save.setText(QCoreApplication.translate("MainWindow", u"\uc800\uc7a5", None))
#if QT_CONFIG(shortcut)
        self.save.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.close.setText(QCoreApplication.translate("MainWindow", u"\ub2eb\uae30", None))
#if QT_CONFIG(shortcut)
        self.close.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+F4", None))
#endif // QT_CONFIG(shortcut)
        self.action.setText(QCoreApplication.translate("MainWindow", u"\ubcf5\uc0ac", None))
        self.action_2.setText(QCoreApplication.translate("MainWindow", u"\ubd99\uc5ec\ub123\uae30", None))
        self.paint.setText("")
        self.img_label.setText("")
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\ud30c\uc77c", None))
    # retranslateUi


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    widget = Ui_MainWindow()
    widget.show()
    sys.exit(app.exec())