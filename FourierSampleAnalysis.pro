#-------------------------------------------------
#
# Project created by QtCreator 2015-12-16T14:18:58
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FourierSampleAnalysis_temp
TEMPLATE = app

OBJECTS_DIR = obj
MOC_DIR = moc

INCLUDEPATH += include

SOURCES += src/*.cpp \
    src/FourierSolver.cpp

HEADERS  += include/*.h \
    include/FourierSolver.h \
    include/float2.h

