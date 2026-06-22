// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("OpenCASCADE");

L = 100;
W = 100;
H = 100;
R = 10;

nL = 10;
nW = 10;
nH = 10;
nR = 15;

Point(1) = {0,  0,  0, 1};
Point(2) = {L,  0,  0, 1};
Point(3) = {L,  W,  0, 1};
Point(4) = {0,  W,  0, 1};
Point(5) = {L/2,  W/2,  0, 0.5};
Point(6) = {L/2+R,  W/2,  0, 5};
Point(7) = {L/2-R,  W/2,  0, 5};




//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Circle(5) = {7, 5, 6};
//+
Circle(6) = {6, 5, 7};
//+
Line(7) = {7, 5};
//+
Line(8) = {5, 6};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Curve Loop(2) = {5, 6};
//+
Plane Surface(1) = {1, 2};
//+
Curve Loop(3) = {6, 7, 8};
//+
Plane Surface(2) = {3};
//+
Curve Loop(4) = {5, -8, -7};
//+
Plane Surface(3) = {4};
//+
Transfinite Curve {4, 3, 2, 1} = 2*Floor(nW/2)+1 Using Progression 1;
//+
Transfinite Curve {-8, 7} = 2*Floor(nR/2)+1 Using Progression 1/1.4;
//+
//+
Extrude {0, 20, H/2} {
  Surface{1}; Surface{3}; Surface{2}; Layers {{8,16,8},{0.05,0.95, 1}}; Recombine;
}
Extrude {0, 0, -H/2} {
  Surface{1}; Surface{3}; Surface{2}; Layers {{8,16,8},{0.05,0.95, 1}}; Recombine;
}
