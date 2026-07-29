// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 50;
W = 73;
H = 45;
ang = 30*Pi/180;
h = H/2;
l = L/2;

nL = 20;
nW = 20;
nH = 20;

Point(1) = {0,  0,  0, 1};
Point(2) = {L,  0,  0, 1};
Point(3) = {L,  W,  0, 1};
Point(4) = {0,  W,  0, 1};
Point(5) = {0,  0,  H, 1};
Point(6) = {L,  0,  H, 1};
Point(7) = {L,  W,  H, 1};
Point(8) = {0,  W,  H, 1};//+
Point(9)  = {0,  0,  h+l*Sin(ang), 1};
Point(10) = {L,  0,  h+l*Sin(ang), 1};
Point(11) = {L,  W,  h-l*Sin(ang), 1};
Point(12) = {0,  W,  h-l*Sin(ang), 1};//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 5};
//+
Line(9) = {9, 10};
//+
Line(10) = {10, 11};
//+
Line(11) = {11, 12};
//+
Line(12) = {12, 9};
//+
Line(13) = {5, 9};
//+
Line(14) = {9, 1};
//+
Line(15) = {8, 12};
//+
Line(16) = {12, 4};
//+
Line(17) = {7, 11};
//+
Line(18) = {11, 3};
//+
Line(19) = {6, 10};
//+
Line(20) = {10, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, 9, 10, 11};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {8, 5, 6, 7};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {7, 15, -11, -17};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {11, 16, -3, -18};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {10, 18, -2, -20};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {6, 17, -10, -19};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {8, 13, -12, -15};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {12, 14, -4, -16};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {5, 19, -9, -13};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {9, 20, -1, -14};
//+
Plane Surface(11) = {11};
//+
Surface Loop(1) = {1, 9, 11, 6, 5, 2};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {3, 8, 10, 7, 4, 2};
//+
Volume(2) = {2};
//+
Transfinite Curve {2, 10, 6, 12, 8, 4} = nW Using Progression 1;
//+
Transfinite Curve {18, 17, 15, 16, 14, 13, 20, 19} = nH/2 Using Progression 1;
//+
Transfinite Curve {5, 9, 1, 3, 11, 7} = nL Using Progression 1;
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Transfinite Surface {1};
//+
Transfinite Surface {11};
//+
Transfinite Surface {10};
//+
Transfinite Surface {3};
//+
Transfinite Surface {8};
//+
Transfinite Surface {9};
//+
Transfinite Surface {1};
//+
Transfinite Surface {6};
//+
Transfinite Surface {7};
//+
Transfinite Surface {3};
//+
Transfinite Surface {2};
//+
Transfinite Volume{1};
//+
Transfinite Volume{2};
