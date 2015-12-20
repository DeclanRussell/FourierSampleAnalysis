#-------------------------------------------------
#
# Project created by QtCreator 2015-12-16T14:18:58
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FourierSampleAnalysis
TEMPLATE = app

OBJECTS_DIR = obj
MOC_DIR = moc

INCLUDEPATH += include

SOURCES += src/*.cpp \
    src/FourierSolver.cpp

HEADERS  += include/*.h \
    include/FourierSolver.h \
    include/float2.h

DEFINES+= USE_PTHREADS

contains(DEFINES,USE_PTHREADS)
{
    # This is the windows version of pthreads. Should be nice and easy to install on any other operating system that isnt windows
    win32:LIBS+= -lpthreadVC2
    unix:LIBS+= -lpthread
    # This flag is needed to be added such that pthreads works
    QMAKE_CXXFLAGS+=-DHAVE_STRUCT_TIMESPEC
    message("Building project with pthreads")
}
