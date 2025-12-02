close all
clear
clc

% Original Grid 
BASEFLOW =readmatrix("../Input/BASEFLOW/BASEFLOW_ND_Re1000k.csv");
PATH     = "../Input"; 

x = BASEFLOW(:, 5); 
y = BASEFLOW(:, 6);

RHO_ND = BASEFLOW(:, 3);
U_ND = BASEFLOW(:, 1);
V_ND = BASEFLOW(:, 2);
T_ND = BASEFLOW(:, 4);

% % Target Grid in4 meshgrid() format, i.e. matrix
Z1 = readmatrix(PATH + "/MESH/StabZ1.csv");
Y1 = readmatrix(PATH + "/MESH/StabY1.csv");

Z2 = readmatrix(PATH + "/MESH/StabZ2.csv");
Y2 = readmatrix(PATH + "/MESH/StabY2.csv");

Z3 = readmatrix(PATH + "/MESH/StabZ3.csv");
Y3 = readmatrix(PATH + "/MESH/StabY3.csv");
% 
% Z4 = readmatrix(PATH + "/StabZ4.csv");
% Y4 = readmatrix(PATH + "/StabY4.csv");
% 
% Z5 = readmatrix(PATH + "MESH/StabZ5.csv");
% Y5 = readmatrix(PATH + "MESH/StabY5.csv");

% griddata() with method "cubic" is a triangulation,
% C2 continous interpolation procedure based on Delaunay triangulation
method = "cubic";

% Interpolation onto stability grids Rho
RHO1 = griddata(x, y, RHO_ND, Z1, Y1, method);
RHO1 = fillmissing2(RHO1, "nearest");
RHO2 = griddata(x, y, RHO_ND, Z2, Y2, method);
RHO2 = fillmissing2(RHO2, "nearest");
RHO3 = griddata(x, y, RHO_ND, Z3, Y3, method);
RHO3 = fillmissing2(RHO3, "nearest");
% RHO4 = griddata(x, y, RHO_ND, Z4, Y4, method);
% RHO4 = fillmissing2(RHO4, "nearest");
% RHO5 = griddata(x, y, RHO_ND, Z5, Y5, method);
% RHO5 = fillmissing2(RHO5, "nearest");

writematrix(RHO1, PATH + "/BASEFLOW/Rho1.csv")
writematrix(RHO2, PATH + "/BASEFLOW/Rho2.csv")
writematrix(RHO3, PATH + "/BASEFLOW/Rho3.csv")
% writematrix(RHO4, PATH + "/Rho4.csv")
% writematrix(RHO5, PATH + "BASEFLOW/Rho5.csv")

% Interpolation onto stability grids U
U1 = griddata(x, y, U_ND, Z1, Y1, method);
U1 = fillmissing2(U1, "nearest");
U2 = griddata(x, y, U_ND, Z2, Y2, method);
U2 = fillmissing2(U2, "nearest");
U3 = griddata(x, y, U_ND, Z3, Y3, method);
U3 = fillmissing2(U3, "nearest");
% U4 = griddata(x, y, U_ND, Z4, Y4, method);
% U4 = fillmissing2(U4, "nearest");
% U5 = griddata(x, y, U_ND, Z5, Y5, method);
% U5 = fillmissing2(U5, "nearest");

writematrix(U1, PATH + "/BASEFLOW/U1.csv")
writematrix(U2, PATH + "/BASEFLOW/U2.csv")
writematrix(U3, PATH + "/BASEFLOW/U3.csv")
% writematrix(U4, PATH + "/U4.csv")
% writematrix(U5, PATH + "BASEFLOW/U5.csv")

% Interpolation onto stability grids V
V1 = griddata(x, y, V_ND, Z1, Y1, method);
V1 = fillmissing2(V1, "nearest");
V2 = griddata(x, y, V_ND, Z2, Y2, method);
V2 = fillmissing2(V2, "nearest");
V3 = griddata(x, y, V_ND, Z3, Y3, method);
V3 = fillmissing2(V3, "nearest");
% V4 = griddata(x, y, V_ND, Z4, Y4, method);
% V4 = fillmissing2(V4, "nearest");
% V5 = griddata(x, y, V_ND, Z5, Y5, method);
% V5 = fillmissing2(V5, "nearest");

writematrix(V1, PATH + "/BASEFLOW/V1.csv")
writematrix(V2, PATH + "/BASEFLOW/V2.csv")
writematrix(V3, PATH + "/BASEFLOW/V3.csv")
% writematrix(V4, PATH + "/V4.csv")
% writematrix(V5, PATH + "BASEFLOW/V5.csv")

% Interpolation onto stability grids T
T1 = griddata(x, y, T_ND, Z1, Y1, method);
T1 = fillmissing2(T1, "nearest");
T2 = griddata(x, y, T_ND, Z2, Y2, method);
T2 = fillmissing2(T2, "nearest");
T3 = griddata(x, y, T_ND, Z3, Y3, method);
T3 = fillmissing2(T3, "nearest");
% T4 = griddata(x, y, T_ND, Z4, Y4, method);
% T4 = fillmissing2(T4, "nearest");
% T5 = griddata(x, y, T_ND, Z5, Y5, method);
% T5 = fillmissing2(T5, "nearest");

writematrix(abs(T1), PATH + "/BASEFLOW/T1.csv")
writematrix(abs(T2), PATH + "/BASEFLOW/T2.csv")
writematrix(abs(T3), PATH + "/BASEFLOW/T3.csv")
% writematrix(abs(T4), PATH + "/T4.csv")
% writematrix(abs(T5), PATH + "BASEFLOW/T5.csv")

% Rlevels = linspace(min(RHO_ND), max(RHO_ND), 100);
% Ulevels = linspace(min(U_ND), max(U_ND), 100);
% Vlevels = linspace(min(V_ND), max(V_ND), 100);
% Tlevels = linspace(min(T_ND), max(T_ND), 200);
% 
% figure(1)
% contourf(Z1, Y1, T1, Tlevels, 'LineColor','none')
% hold on
% contourf(Z2, Y2, T2, Tlevels,'LineColor','none')
% contourf(Z3, Y3, T3, Tlevels, 'LineColor','none')
% % contourf(Z4, Y4, T4, Tlevels, 'LineColor','none')
% % contourf(Z5, Y5, T5, Tlevels, 'LineColor','none')
% title("Temperature")
% hold off
% 
% figure(2)
% contourf(Z1, Y1, U1, Ulevels, 'LineColor','none')
% hold on
% contourf(Z2, Y2, U2, Ulevels, 'LineColor','none')
% contourf(Z3, Y3, U3, Ulevels, 'LineColor','none')
% % contourf(Z4, Y4, U4, Ulevels, 'LineColor','none')
% % contourf(Z5, Y5, U5, Ulevels, 'LineColor','none')
% title("U Vel")
% hold off
% 
% figure(3)
% contourf(Z1, Y1, V1, Vlevels, 'LineColor','none')
% hold on
% contourf(Z2, Y2, V2, Vlevels,  'LineColor','none')
% contourf(Z3, Y3, V3, Vlevels,  'LineColor','none')
% % contourf(Z4, Y4, V4, Vlevels,  'LineColor','none')
% % contourf(Z5, Y5, V5, Vlevels,  'LineColor','none')
% title("V Vel")
% hold off
% 
% figure(4)
% contourf(Z1, Y1, RHO1, Rlevels,  'LineColor','none')
% hold on
% contourf(Z2, Y2, RHO2, Rlevels, 'LineColor','none')
% contourf(Z3, Y3, RHO3, Rlevels, 'LineColor','none')
% % contourf(Z4, Y4, RHO4, Rlevels, 'LineColor','none')
% % contourf(Z5, Y5, RHO5, Rlevels, 'LineColor','none')
% title("Density")
% hold off
% 
% set(gca,'DataAspectRatio',[1 1 1])
% 
% % surf(Z1, Y1, ones(length(Y1(:,1)), length(Z1(1,:))))
% % hold on
% % surf(Z2, Y2, ones(length(Y2(:,1)), length(Z2(1,:))))
% % surf(Z3, Y3, ones(length(Y3(:,1)), length(Z3(1,:))))
% % surf(Z4, Y4, ones(length(Y4(:,1)), length(Z4(1,:))))
% % surf(Z5, Y5, ones(length(Y5(:,1)), length(Z5(1,:))))
% % hold off


