// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 73;
H = 43;

nL = 3;
nW = 3;
nH = 10;
S = L/10;

Point(1) = {0,  0,  0, S};
Point(2) = {0,  0,  H, S};
Point(3) = {0,  W,  H, S};
Point(4) = {0,  W,  0, S};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Extrude {43, 0, 0} {
  Surface{1}; Layers {5}; Recombine;
}
