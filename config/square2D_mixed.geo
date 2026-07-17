// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 75;
H = 45;

Lf = 22;
Disturb = 0.2;
L2 = 5;

nL = 25;
nW = 7;
nH = 7;

Point(1) = {0,  0,  0, nL};
Point(2) = {L,  0,  0, nL};
Point(3) = {L,  W,  0, nL};
Point(4) = {0,  W,  0, nL};
Point(5) = {L/2, W/2, 0, nL};
Point(6) = {L/2, W/2-L2, 0, nL};
Point(7) = {L/2 + Disturb, W/2+Lf,0,nL};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {1, 6};
//+
Line(6) = {6, 5};
//+
Line(7) = {5, 3};
//+
Line(8) = {7, 4};
//+
Line(9) = {7, 3};
//+
Curve Loop(1) = {9, 3, -8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, -7, -6, -5, 1};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, 5, 6, 7, -9, 8};
//+
Plane Surface(3) = {3};
