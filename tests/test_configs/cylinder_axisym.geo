// Thick-walled cylinder (axisymmetric r-z section), graded toward the inner wall
a = 3;
b = 54;
H = 6;
nr = 41;
grading = 1.15;

Point(1) = {a, 0, 0};
Point(2) = {b, 0, 0};
Point(3) = {b, H, 0};
Point(4) = {a, H, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Transfinite Curve {1} = nr Using Progression grading;
Transfinite Curve {3} = nr Using Progression 1/grading;
Transfinite Curve {2, 4} = 3;
Transfinite Surface {1};
Recombine Surface {1};
Physical Surface(1) = {1};
