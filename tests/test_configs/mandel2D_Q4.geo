// Uniform Q4 rectangle
Lx = 10;
Ly = 2;
nx = 41;
ny = 3;

Point(1) = {0, 0, 0};
Point(2) = {Lx, 0, 0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0, Ly, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Curve {1, 3} = nx;
Transfinite Curve {2, 4} = ny;
Transfinite Surface {1};
Recombine Surface {1};
Physical Surface(1) = {1};
