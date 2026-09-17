// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 75;
H = 45;

nL = 3;
nW = 1;
nH = 7;

Point(1) = {-L,  0,  0, nL};
Point(2) = {2*L,  0,  0, nL};
Point(3) = {2*L,  W,  0, nL};
Point(4) = {-L,  W,  0, nL};
Circle(5) = {L/2, W/3, 0, 10, 0, 2*Pi};
//+
Circle(6) = {2*L/3, W/2+1, 0, 5, 0, 2*Pi};
//+
Point(7) = {0, 55, 0, nL};
Point(8) = {-L, 55, 0, nL};


//+
Line(7) = {4, 8};
//+
Line(8) = {8, 7};
//+
Line(9) = {7, 3};
//+
Line(10) = {3, 4};
//+
Line(11) = {8, 1};
//+
Line(12) = {1, 2};
//+
Line(13) = {2, 3};
//+
Curve Loop(1) = {5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {9, 10, 7, 8};
//+
Plane Surface(3) = {3};

//+
Curve Loop(4) = {12, 13, -9, -8, 11};
//+
Curve Loop(5) = {6};
//+
Curve Loop(6) = {5};
//+
Plane Surface(4) = {4, 5, 6};
//+
MeshSize {5, 6} = nW;
