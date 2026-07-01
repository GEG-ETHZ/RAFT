// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 51;
W = 73;
H = 43;

nL = 2;
nW = 1;
nH = 15;

Point(1) = {0,  0,  0, 1};
Point(2) = {L,  0,  0, 1};
Point(3) = {L,  W,  0, 1};
Point(4) = {0,  W,  0, 1};
Point(5) = {0,  0,  H, 1};
Point(6) = {L,  0,  H, 1};
Point(7) = {L,  W,  H, 1};
Point(8) = {0,  W,  H, 1};//+
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
Line(13) = {6, 8};
//+
Line(14) = {2, 4};
//+
Curve Loop(1) = {5, -13, 12};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 13, 6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {1, -14, 4};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {3, 14, 2};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {10, -7, -11, 3};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {10, 12, -8, -4};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {1, 9, -5, -8};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {2, 11, -6, -9};
//+
Plane Surface(8) = {8};
//+
//+
Curve Loop(9) = {9, -13, -10, 14};
//+
Plane Surface(9) = {9};
//+
//+
Transfinite Curve {6, 7, 13, 5, 12, 14, 2, 3, 4, 1} = nL+1 Using Progression 1;
//+
Transfinite Curve {9, 11, -10, -8} = nH+1 Using Progression 1.1;
//+
Transfinite Surface {2};
//+
Transfinite Surface {1};
//+
Transfinite Surface {7};
//+
Transfinite Surface {8};
//+
Transfinite Surface {5};
//+
Transfinite Surface {6};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
Transfinite Surface {9};

//+
Surface Loop(2) = {1, 7, 3, 6, 9};
//+
Volume(1) = {2};
//+
Surface Loop(3) = {9, 2, 5, 8, 4};
//+
Volume(2) = {3};
//+
Transfinite Volume{1, 2};
