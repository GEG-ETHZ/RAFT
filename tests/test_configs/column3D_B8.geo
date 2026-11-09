// Uniform B8 column
Lx = 1;
Ly = 1;
Lz = 10;
nz = 40;

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
Transfinite Curve {1, 2, 3, 4} = 2;
Transfinite Surface {1};
Recombine Surface {1};
Extrude {0, 0, Lz} { Surface{1}; Layers{nz}; Recombine; }
Physical Volume(1) = {1};
