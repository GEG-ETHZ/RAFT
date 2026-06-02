// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 73;
H = 43;

nL = 1;
nW = 1;
nH = 10;

Point(1) = {0,  0,  0, 1};
Point(4) = {L,  0,  0, 1};
Point(8) = {L,  W,  0, 1};
Point(5) = {0,  W,  0, 1};
Point(2) = {0,  0,  H, 1};
Point(3) = {L,  0,  H, 1};
Point(7) = {L,  W,  H, 1};
Point(6) = {0,  W,  H, 1};//+
Line(1) = {1, 4};
//+
Line(2) = {4, 3};
//+
Line(3) = {3, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {5, 8};
//+
Line(6) = {8, 7};
//+
Line(7) = {7, 6};
//+
Line(8) = {1, 5};
//+
Line(9) = {4, 8};
//+
Line(10) = {2, 6};
//+
Line(11) = {3, 7};
//+
Line(12) = {6, 5};
//+
Curve Loop(1) = {6, -11, -2, 9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {7, -10, -3, 11};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {12, 5, 6, 7};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {8, -12, -10, 4};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {5, -9, -1, 8};
//+
Plane Surface(6) = {6};
//+
Surface Loop(1) = {3, 4, 5, 6, 1, 2};
//+
Volume(1) = {1};
//+
Transfinite Curve {1, 5, 7, 3} = nL+1 Using Progression 1;
//+
Transfinite Curve {-8, 9, -11, 10} = nH+1 Using Progression 1;
//+
Transfinite Curve {12, 6, 2, 4} = nW+1 Using Progression 1;
//+
Transfinite Surface {6};
//+
Transfinite Surface {3};
//+
Transfinite Surface {2};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Transfinite Surface {1};
//+
Transfinite Volume{1};
