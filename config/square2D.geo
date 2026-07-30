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

Point(7) = {0, 55, 0, nL};
Point(8) = {-L, 55, 0, nL};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};

//+
Circle(5) = {L/2, W/3, 0, 10, 0, 2*Pi};
//+
Circle(6) = {2*L/3, W/2+1, 0, 5, 0, 2*Pi};
//+
Curve Loop(4) = {6};
//+
Curve Loop(5) = {5};
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
//+
Line(7) = {8, 7};
//+
Line(8) = {7, 3};
//+
Curve Loop(6) = {1, 2, 3, 4};
//+
Line(9) = {1, 8};
//+
Line(10) = {8, 4};
//+
Curve Loop(6) = {3, -10, 7, 8};
//+
Plane Surface(1) = {6};
//+
Curve Loop(7) = {9, 7, 8, -2, -1};
//+
Curve Loop(8) = {6};
//+
Curve Loop(9) = {6};
//+
Curve Loop(10) = {5};
//+//+
Curve Loop(11) = {6};
//+
Plane Surface(2) = {11};
//+
Curve Loop(12) = {5};
//+
Plane Surface(3) = {12};
//+
//+
Curve Loop(13) = {8, -2, -1, 9, 7};
//+
Curve Loop(14) = {5};
//+
Curve Loop(15) = {6};
//+
Plane Surface(4) = {13, 14, 15};
//+
MeshSize {10, 9} = nL;
