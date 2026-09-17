// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 250;
W = 730;
H = 450;
ang = 30*Pi/180;

H2 = 100;
h = H/2;
l = L/2;
w = W/2;

nL = 50;
nW = 20;
nH = 2;

Point(1) = {0,  0,  0, nL};
Point(2) = {L,  0,  0, nL};
Point(3) = {L,  W,  0, nL};
Point(4) = {0,  W,  0, nL};
Point(5) = {0,  0,  H, nL};
Point(6) = {L,  0,  H, nL};
Point(7) = {L,  W,  H, nL};
Point(8) = {0,  W,  H, nL};//+
Point(9)  = {0,  0, H-H2,  nL};
Point(10) = {L,  0, H-H2, nL};
Point(11) = {L,  W,  H2, nL};
Point(12) = {0,  W,  H2, nL};//+

Point(13) = {0, 550, 250, nH};

R = 50;

Point(14) = {0, W/2, 0, nW};
Point(15) = {0, W/2+R, 0, nW};
Point(16) = {0, W/2-R, 0, nW};

Point(17) = {0, W/2, H, nW};
Point(18) = {0, W/2+R, H, nW};
Point(19) = {0, W/2-R, H, nW};

y = R/w*(h-H2);
Point(20) = {0, W/2, h, nW};
Point(21) = {0, W/2+R, h-y, nW};
Point(22) = {0, W/2-R, h+y, nW};//+

Point(23) = {0, 152, 340, nH};

Line(1) = {3, 4};
//+
Line(2) = {4, 15};
//+
Line(3) = {15, 14};
//+
Line(4) = {14, 16};
//+
Line(5) = {16, 1};
//+
Line(6) = {1, 2};
//+
Line(7) = {2, 3};
//+
Line(8) = {4, 12};
//+
Line(9) = {12, 11};
//+
Line(10) = {11, 3};
//+
Line(11) = {12, 8};
//+
Line(12) = {7, 8};
//+
Line(13) = {11, 7};
//+
Line(14) = {1, 9};
//+
Line(15) = {9, 5};
//+
Line(16) = {5, 19};
//+
Line(17) = {19, 17};
//+
Line(18) = {17, 18};
//+
Line(19) = {18, 8};
//+
Line(20) = {5, 6};
//+
Line(21) = {6, 7};
//+
Line(22) = {2, 10};
//+
Line(23) = {10, 6};
//+
Line(24) = {9, 10};
//+
Line(25) = {10, 11};
//+
//+
//+
Line(26) = {12, 21};
//+
Line(27) = {21, 20};
//+
Line(28) = {20, 22};
//+
Line(29) = {22, 9};
//+
Line(30) = {18, 21};
//+
Line(31) = {21, 15};
//+
Line(32) = {22, 19};
//+
Line(33) = {16, 22};
//+
Line(34) = {18, 13};
//+
Line(35) = {13, 12};
//+
Curve Loop(1) = {34, 35, 11, -19};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {26, -30, 34, 35};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {32, 17, 18, 30, 27, 28};
//+
Plane Surface(3) = {3};
//+
Curve Loop(5) = {5, 14, -29, -33};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {33, -28, -27, 31, 3, 4};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {2, -31, -26, -8};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {1, 2, 3, 4, 5, 6, 7};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {14, 24, -22, -6};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {24, 23, -20, -15};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {11, -12, -13, -9};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {8, 9, 10, 1};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {25, 13, -21, -23};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {22, 25, 10, -7};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {21, 12, -19, -18, -17, -16, 20};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {29, 24, 25, -9, 26, 27, 28};
//+
Plane Surface(16) = {16};
//+

Line(36) = {19, 23};
//+
Line(37) = {23, 9};
//+
Curve Loop(17) = {32, 36, 37, -29};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {36, 37, 15, 16};
//+
Plane Surface(18) = {18};
//
Surface Loop(1) = {2, 3, 10, 13, 11, 1, 15, 16, 17, 18};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {14, 9, 5, 8, 12, 7, 6, 16};
//+
Volume(2) = {2};
//+
