#include "mainwindow.h"

#include <QPushButton>
#include <QPushButton>
#include <QFileDialog>
#include <QFileInfo>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPixmap>
#include <iostream>
#include <iostream>

//----------------------------------------------------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setWindowTitle("Fourier Analysis");
    resize(800,900);
    m_centralWgt = new QGroupBox(this);
    this->setCentralWidget(m_centralWgt);
    m_UI = new QGridLayout(m_centralWgt);
    m_centralWgt->setLayout(m_UI);
    m_fourierSolver = new FourierSolver;
    m_psLbl = new QLabel(m_centralWgt);
    m_UI->addWidget(m_psLbl,0,0,1,2);

    // Create a button to import 2D points from a file
    QPushButton *resizeImg = new QPushButton("Resize image to window",m_centralWgt);
    m_UI->addWidget(resizeImg,1,0,1,2);
    connect(resizeImg,SIGNAL(pressed()),this,SLOT(resizeImage()));

    // Create a field for our range selection
    m_UI->addWidget(new QLabel("Range Selection",m_centralWgt),2,0,1,1);
    QDoubleSpinBox *rangeselSpn = new QDoubleSpinBox(m_centralWgt);
    rangeselSpn->setValue(m_fourierSolver->getRangeSelection());
    rangeselSpn->setDecimals(4);
    connect(rangeselSpn,SIGNAL(valueChanged(double)),this,SLOT(setRangeSelection(double)));
    m_UI->addWidget(rangeselSpn,2,1,1,1);

    // Create a field for our gaussian standar deviation
    m_UI->addWidget(new QLabel("Gaussian Standard Deviation",m_centralWgt),3,0,1,1);
    QDoubleSpinBox *sdSpn = new QDoubleSpinBox(m_centralWgt);
    sdSpn->setMaximum(INFINITY);
    sdSpn->setDecimals(4);
    sdSpn->setValue(m_fourierSolver->getStandardDeviation());
    connect(sdSpn,SIGNAL(valueChanged(double)),this,SLOT(setSD(double)));
    m_UI->addWidget(sdSpn,3,1,1,1);

    // Create a button to import 2D points from a file
    QPushButton *imp2D = new QPushButton("Import 2D Points from File",m_centralWgt);
    m_UI->addWidget(imp2D,4,0,1,2);
    connect(imp2D,SIGNAL(pressed()),this,SLOT(import2DData()));

    // Create a button to import our differentials from a file
    QPushButton *impdiff = new QPushButton("Import differentials from File",m_centralWgt);
    m_UI->addWidget(impdiff,5,0,1,2);
    connect(impdiff,SIGNAL(pressed()),this,SLOT(importDiffData()));

    // Create a button to analys our data
    QPushButton *anylBtn = new QPushButton("Analyse Data",m_centralWgt);
    m_UI->addWidget(anylBtn,6,0,1,2);
    connect(anylBtn,SIGNAL(pressed()),this,SLOT(analyse()));

    // Create a save button
    QPushButton *saveBtn = new QPushButton("Save Image",m_centralWgt);
    m_UI->addWidget(saveBtn,7,0,1,2);
    connect(saveBtn,SIGNAL(pressed()),this,SLOT(saveImage()));


}
//----------------------------------------------------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    delete m_fourierSolver;
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::import2DData()
{
    QString dir = QFileDialog::getOpenFileName(this,"Import 2D Points from File");
    if(!dir.isEmpty())
    {
        m_fourierSolver->import2DFromFile(dir);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::importDiffData()
{
    QString dir = QFileDialog::getOpenFileName(this,"Import Differentials from File");
    if(!dir.isEmpty())
    {
        m_fourierSolver->importDifferentialsFromFile(dir);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::analyse()
{
    m_fourierSolver->analysePoints();
    QPixmap img = QPixmap::fromImage(m_fourierSolver->getPSImage());
    m_psLbl->setPixmap(img.scaled(m_psLbl->width(),m_psLbl->height(),Qt::KeepAspectRatio));
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::saveImage()
{
    QString loc =  QFileDialog::getSaveFileName(this,"Save Power Spectrum Image");
    if(!loc.endsWith(".png",Qt::CaseInsensitive)) loc+=".png";
    m_fourierSolver->getPSImage().save(loc,"PNG");
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::resizeImage()
{
    QPixmap img = QPixmap::fromImage(m_fourierSolver->getPSImage());
    m_psLbl->setPixmap(img.scaled(m_psLbl->width(),m_psLbl->height(),Qt::KeepAspectRatio));
}
//----------------------------------------------------------------------------------------------------------------------
