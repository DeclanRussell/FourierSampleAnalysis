#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGroupBox>
#include <QGridLayout>
#include <QLabel>

#include "FourierSolver.h"


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to import 2D data to our fourier analysis class
    //----------------------------------------------------------------------------------------------------------------------
    void import2DData();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief call to analyse and show our imported data
    //----------------------------------------------------------------------------------------------------------------------
    void analyse();
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief central widget of our main window
    //----------------------------------------------------------------------------------------------------------------------
    QGroupBox *m_centralWgt;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our grid laout of our central widget
    //----------------------------------------------------------------------------------------------------------------------
    QGridLayout *m_UI;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief class to perfrom our fourier analysis
    //----------------------------------------------------------------------------------------------------------------------
    FourierSolver *m_fourierSolver;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief labal to show the image of our power spectrum once created
    //----------------------------------------------------------------------------------------------------------------------
    QLabel *m_psLbl;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // MAINWINDOW_H
