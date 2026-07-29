// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 75;
H = 45;

nL = 4;
nW = 7;
nH = 7;

Point(1) = {-L,  0,  0, nL};
Point(2) = {2*L,  0,  0, nL};
Point(3) = {2*L,  W,  0, nL};
Point(4) = {-L,  W,  0, nL};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Circle(5) = {L/2, W/3, 0, 10, 0, 2*Pi};
//+
Circle(6) = {2*L/3, W/2+1, 0, 5, 0, 2*Pi};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Curve Loop(2) = {5};
//+
Curve Loop(3) = {6};
//+
Plane Surface(1) = {1, 2, 3};
//+
Curve Loop(4) = {6};
//+
Curve Loop(5) = {5};
//+
Plane Surface(2) = {4, 5};
//+
MeshSize {5, 6} = nL/2;
//+
Curve Loop(7) = {6};
//+
Plane Surface(3) = {7};
//+
Field[1] = Cylinder;
//+
Field[1].VIn = 0.2;
//+
Field[1].XCenter = 40;
//+
Field[1].YCenter = 41;
//+
Field[1].Radius = 2;
//+
Background Field = 1;
//+
Field[1].Radius = 5;
//+
Field[1].VIn = 0.5;
//+
Field[1].VIn = 0.3;
