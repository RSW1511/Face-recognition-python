# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'register.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import mysql.connector as con


class Ui_RegisterMainWindow(object):
    def insert_data(self):
        from login import Ui_loginMainWindow
        try:
            db = con.connect(host="localhost", user="root", password="", database="studentdb")
            c = db.cursor()

            Name = self.Name_edit.text()
            Username = self.UserName_edit.text()
            Password = self.Pass_edit.text()
            Email = self.Email_edit.text()
            Mobile = self.Mobile_edit.text()
            Designation = self.desig_edit.text()

            query = "INSERT INTO exp1 (Name, Username, Password, Email, Mobile, Designation) VALUES(%s,%s,%s,%s,%s,%s)"
            value = (Name, Username, Password, Email, Mobile, Designation)

            c.execute(query, value)
            db.commit()
            self.window = QtWidgets.QMainWindow()
            self.ui = Ui_loginMainWindow()
            self.ui.setupUi(self.window)
            self.window.show()

        except con.Error as e:
            self.labelResult.setText("Unsuccessful")
    def openWindowlogin(self):
        from login import Ui_loginMainWindow
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_loginMainWindow()
        self.ui.setupUi(self.window)
        self.window.show()
    def setupUi(self, RegisterMainWindow):
        RegisterMainWindow.setObjectName("RegisterMainWindow")
        RegisterMainWindow.resize(1093, 801)
        self.centralwidget = QtWidgets.QWidget(RegisterMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(-270, -20, 1371, 821))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(260, 20, 1111, 811))
        self.label.setStyleSheet("image: url(:/register/register.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.Name_edit = QtWidgets.QLineEdit(self.frame)
        self.Name_edit.setGeometry(QtCore.QRect(550, 340, 221, 41))
        self.Name_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.Name_edit.setObjectName("Name_edit")
        self.Email_edit = QtWidgets.QLineEdit(self.frame)
        self.Email_edit.setGeometry(QtCore.QRect(550, 430, 221, 41))
        self.Email_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.Email_edit.setObjectName("Email_edit")
        self.UserName_edit = QtWidgets.QLineEdit(self.frame)
        self.UserName_edit.setGeometry(QtCore.QRect(810, 340, 221, 41))
        self.UserName_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.UserName_edit.setObjectName("UserName_edit")
        self.Pass_edit = QtWidgets.QLineEdit(self.frame)
        self.Pass_edit.setGeometry(QtCore.QRect(810, 430, 221, 41))
        self.Pass_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.Pass_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.Pass_edit.setObjectName("Pass_edit")
        self.Mobile_edit = QtWidgets.QLineEdit(self.frame)
        self.Mobile_edit.setGeometry(QtCore.QRect(550, 510, 221, 41))
        self.Mobile_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.Mobile_edit.setObjectName("Mobile_edit")
        self.desig_edit = QtWidgets.QLineEdit(self.frame)
        self.desig_edit.setGeometry(QtCore.QRect(810, 510, 221, 41))
        self.desig_edit.setStyleSheet("font: 75 12pt \"Georgia\";")
        self.desig_edit.setObjectName("desig_edit")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(700, 570, 221, 20))
        self.checkBox.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 9pt \"MS Shell Dlg 2\";")
        self.checkBox.setObjectName("checkBox")
        self.sign_button = QtWidgets.QPushButton(self.frame,clicked=lambda: self.insert_data())
        self.sign_button.setGeometry(QtCore.QRect(710, 660, 161, 51))
        self.sign_button.setStyleSheet("color: rgb(0, 0, 77);\n"
"font: 75 12pt \"Georgia\";")
        self.sign_button.setObjectName("sign_button")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(550, 310, 91, 21))
        self.label_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(810, 310, 101, 21))
        self.label_3.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(810, 400, 91, 21))
        self.label_4.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setGeometry(QtCore.QRect(550, 400, 91, 21))
        self.label_5.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setGeometry(QtCore.QRect(550, 480, 91, 21))
        self.label_6.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setGeometry(QtCore.QRect(810, 480, 121, 31))
        self.label_7.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setGeometry(QtCore.QRect(720, 60, 211, 31))
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 75 18pt \"Georgia\";")
        self.label_8.setObjectName("label_8")
        RegisterMainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(RegisterMainWindow)
        QtCore.QMetaObject.connectSlotsByName(RegisterMainWindow)

    def retranslateUi(self, RegisterMainWindow):
        _translate = QtCore.QCoreApplication.translate
        RegisterMainWindow.setWindowTitle(_translate("RegisterMainWindow", "RegisterMainWindow"))
        self.checkBox.setText(_translate("RegisterMainWindow", "Agree to terms and conditions"))
        self.sign_button.setText(_translate("RegisterMainWindow", "SIGN UP"))
        self.label_2.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Name</span></p></body></html>"))
        self.label_3.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">User Name</span></p><p><br/></p></body></html>"))
        self.label_4.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Password</span></p><p><br/></p></body></html>"))
        self.label_5.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Email Id</span></p><p><br/></p></body></html>"))
        self.label_6.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Mobile No</span></p></body></html>"))
        self.label_7.setText(_translate("RegisterMainWindow", "<html><head/><body><p><span style=\" color:#ffffff;\">Designation</span></p><p><br/></p></body></html>"))
        self.label_8.setText(_translate("RegisterMainWindow", "Criminal Eye"))
import register_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RegisterMainWindow = QtWidgets.QMainWindow()
    ui = Ui_RegisterMainWindow()
    ui.setupUi(RegisterMainWindow)
    RegisterMainWindow.show()
    sys.exit(app.exec_())
