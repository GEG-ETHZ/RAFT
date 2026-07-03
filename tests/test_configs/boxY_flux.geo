// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 73;
H = 43;

nL = 1;
nW = 4;
nH = 10;

Point(1) = {0,  0,  0, nW};
Point(2) = {L,  0,  0, nW};
Point(3) = {L,  0,  H, nW};
Point(4) = {0,  0,  H, nW};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Extrude {0, W, 0} {
  Surface{1}; Layers {5}; Recombine;
}
