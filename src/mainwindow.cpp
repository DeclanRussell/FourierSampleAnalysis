#include "mainwindow.h"

#include <QPushButton>
#include <QPushButton>
#include <QFileDialog>
#include <QPixmap>
#include <iostream>

//----------------------------------------------------------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setWindowTitle("Fourier Analysis");
    resize(500,500);
    m_centralWgt = new QGroupBox(this);
    this->setCentralWidget(m_centralWgt);
    m_UI = new QGridLayout(m_centralWgt);
    m_centralWgt->setLayout(m_UI);
    m_fourierSolver = new FourierSolver;
    m_psLbl = new QLabel(m_centralWgt);
    m_UI->addWidget(m_psLbl,0,0,1,1);

    // Create a button to import our data
    QPushButton *imp2D = new QPushButton("Import 2D Points from File",m_centralWgt);
    m_UI->addWidget(imp2D,1,0,1,1);
    connect(imp2D,SIGNAL(pressed()),this,SLOT(import2DData()));

    // Create a button to analys our data
    QPushButton *anylBtn = new QPushButton("Analyse Data",m_centralWgt);
    m_UI->addWidget(anylBtn,2,0,1,1);
    connect(anylBtn,SIGNAL(pressed()),this,SLOT(analyse()));


}
//----------------------------------------------------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    delete m_fourierSolver;
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::import2DData()
{
    m_fourierSolver->import2DFromFile(QFileDialog::getOpenFileName(this,"Import 2D Points from File"));
}
//----------------------------------------------------------------------------------------------------------------------
void MainWindow::analyse()
{
    m_fourierSolver->analysePoints();
    QPixmap img = QPixmap::fromImage(m_fourierSolver->getPSImage());
    m_psLbl->setPixmap(img.scaled(m_psLbl->width(),m_psLbl->height(),Qt::KeepAspectRatio));
}
//----------------------------------------------------------------------------------------------------------------------
