// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 75;
H = 45;

nL = 10;
nW = 7;
nH = 7;

Point(1) = {0,  0,  0, nL};
Point(2) = {L,  0,  0, nL};
Point(3) = {L,  W,  0, nL};
Point(4) = {0,  W,  0, nL};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
