close all
clear
clc

%% PATH
PATH = "/Users/vojtechpezlar/Dropbox/PhD/Development/Python/Tachyon_HPC/Python_Code/";

%% DATA
X1 = readmatrix(PATH + "StabZ1.csv");
X2 = readmatrix(PATH + "StabZ2.csv");
X3 = readmatrix(PATH + "StabZ3.csv");
X4 = readmatrix(PATH + "StabZ4.csv");
X5 = readmatrix(PATH + "StabZ5.csv");

Y1 = readmatrix(PATH + "StabY1.csv");
Y2 = readmatrix(PATH + "StabY2.csv");
Y3 = readmatrix(PATH + "StabY3.csv");
Y4 = readmatrix(PATH + "StabY4.csv");
Y5 = readmatrix(PATH + "StabY5.csv");

F1 = readmatrix(PATH + "Sol1.csv");
F2 = readmatrix(PATH + "Sol2.csv");
F3 = readmatrix(PATH + "Sol3.csv");
F4 = readmatrix(PATH + "Sol4.csv");
F5 = readmatrix(PATH + "Sol5.csv");

%% Plotting
surf(X1, Y1, F1)
hold on
surf(X2, Y2, F2)
surf(X3, Y3, F3)
surf(X4, Y4, F4)
surf(X5, Y5, F5)