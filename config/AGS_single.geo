// Gmsh project created on Wed Apr 29 11:24:58 2026
SetFactory("Built-in");

D = 3000; // Lateral Depth
L = 8000; // Lateral Length
W = 100;  // Domain width

b = 0.02; // Domain buffer
R = 30; // Refined Radius

X = L*((1+2*b));
Y = W;
Z = D*(1+b);

X1 = b*L;
X2 = X - b*L;

s_outer = 200;
a  = 0.25*R;          // half-width of inner square
c  = R/Sqrt(2);


nrf = 6; // Number of refined elements
nrl = 40; // Number of lateral elements
nrv = 20; // Number of vertical elements
nrd = 2; // Number of diagonal elements
g = 1.5; // diagonal growth rate

// Make Box
Point(1) = {0,  0,  0, s_outer};
Point(2) = {X,  0,  0, s_outer};
Point(3) = {X,  Y,  0, s_outer};
Point(4) = {0,  Y,  0, s_outer};
Point(5) = {0,  0,  Z, s_outer};
Point(6) = {X,  0,  Z, s_outer};
Point(7) = {X,  Y,  Z, s_outer};
Point(8) = {0,  Y,  Z, s_outer};//+

// Make Lateral Refined box
// Start
Point(9) = {X1-R, Y/2-R, 0, 1};
Point(10) = {X1+R, Y/2-R, 0, 1};
Point(11) = {X1-R, Y/2+R, 0, 1};
Point(12) = {X1+R, Y/2+R, 0, 1};

// End
Point(13) = {X2-R, Y/2-R, 0, 1};
Point(14) = {X2+R, Y/2-R, 0, 1};
Point(15) = {X2-R, Y/2+R, 0, 1};
Point(16) = {X2+R, Y/2+R, 0, 1};

// First Corner
Point(17) = {X1-R, Y/2-R, D-R, 1};
Point(18) = {X1+R, Y/2-R, D-R, 1};
Point(19) = {X1-R, Y/2+R, D-R, 1};
Point(20) = {X1+R, Y/2+R, D-R, 1};
Point(21) = {X1-R, Y/2-R, D+R, 1};
Point(22) = {X1+R, Y/2-R, D+R, 1};
Point(23) = {X1-R, Y/2+R, D+R, 1};
Point(24) = {X1+R, Y/2+R, D+R, 1};

// Second Corner
Point(25) = {X2-R, Y/2-R, D-R, 1};
Point(26) = {X2+R, Y/2-R, D-R, 1};
Point(27) = {X2-R, Y/2+R, D-R, 1};
Point(28) = {X2+R, Y/2+R, D-R, 1};
Point(29) = {X2-R, Y/2-R, D+R, 1};
Point(30) = {X2+R, Y/2-R, D+R, 1};
Point(31) = {X2-R, Y/2+R, D+R, 1};
Point(32) = {X2+R, Y/2+R, D+R, 1};

// Project to outer edge
DX = X1;
DY = Y/2 - R;
DZ = 2*b*D;

// Top and Bottom Faces
Point(33) = {X1 + DX, 0, 0, 1};
Point(34) = {X1 + DX, Y, 0, 1};
Point(35) = {X1 + DX, 0, Z, 1};
Point(36) = {X1 + DX, Y, Z, 1};

Point(37) = {X2 - DX, 0, 0, 1};
Point(38) = {X2 - DX, Y, 0, 1};
Point(39) = {X2 - DX, 0, Z, 1};
Point(40) = {X2 - DX, Y, Z, 1};

// Mid Level
Point(41) = {X1 + DX, 0, Z-DZ, 1};
Point(42) = {X1 + DX, Y, Z-DZ, 1};
Point(43) = {X1 - DX, 0, Z-DZ, 1};
Point(44) = {X1 - DX, Y, Z-DZ, 1};

Point(45) = {X2 + DX, 0, Z-DZ, 1};
Point(46) = {X2 + DX, Y, Z-DZ, 1};
Point(47) = {X2 - DX, 0, Z-DZ, 1};
Point(48) = {X2 - DX, Y, Z-DZ, 1};

//+
Line(1) = {3, 2};
//+
Line(2) = {2, 37};
//+
Line(3) = {37, 38};
//+
Line(4) = {38, 3};
//+
Line(5) = {16, 14};
//+
Line(6) = {14, 13};
//+
Line(7) = {13, 15};
//+
Line(8) = {15, 16};
//+
Line(9) = {38, 15};
//+
Line(10) = {37, 13};
//+
Line(11) = {2, 14};
//+
Line(12) = {3, 16};
//+
Line(13) = {34, 33};
//+
Line(14) = {33, 1};
//+
Line(15) = {1, 4};
//+
Line(16) = {4, 34};
//+
Line(17) = {34, 12};
//+
Line(18) = {33, 10};
//+
Line(19) = {10, 12};
//+
Line(20) = {12, 11};
//+
Line(21) = {11, 9};
//+
Line(22) = {9, 10};
//+
Line(23) = {4, 11};
//+
Line(24) = {1, 9};
//+
Line(25) = {28, 26};
//+
Line(26) = {26, 25};
//+
Line(27) = {25, 27};
//+
Line(28) = {27, 28};
//+
Line(29) = {32, 30};
//+
Line(30) = {30, 29};
//+
Line(31) = {29, 31};
//+
Line(32) = {31, 32};
//+
Line(33) = {32, 28};
//+
Line(34) = {31, 27};
//+
Line(35) = {30, 26};
//+
Line(36) = {29, 25};
//+
Line(37) = {46, 45};
//+
Line(38) = {45, 47};
//+
Line(39) = {47, 48};
//+
Line(40) = {48, 46};
//+
Line(41) = {7, 6};
//+
Line(42) = {6, 39};
//+
Line(43) = {39, 40};
//+
Line(44) = {40, 7};
//+
Line(45) = {7, 46};
//+
Line(46) = {6, 45};
//+
Line(47) = {39, 47};
//+
Line(48) = {40, 48};
//+
Line(49) = {46, 28};
//+
Line(50) = {7, 32};
//+
Line(51) = {6, 30};
//+
Line(52) = {39, 29};
//+
Line(53) = {40, 31};
//+
Line(54) = {47, 25};
//+
Line(55) = {45, 26};
//+
Line(56) = {48, 27};
//+
Line(57) = {20, 18};
//+
Line(58) = {18, 17};
//+
Line(59) = {17, 19};
//+
Line(60) = {19, 20};
//+
Line(61) = {24, 22};
//+
Line(62) = {22, 21};
//+
Line(63) = {21, 23};
//+
Line(64) = {23, 24};
//+
Line(65) = {24, 20};
//+
Line(66) = {18, 22};
//+
Line(67) = {21, 17};
//+
Line(68) = {23, 19};
//+
Line(69) = {36, 35};
//+
Line(70) = {35, 5};
//+
Line(71) = {5, 8};
//+
Line(72) = {8, 36};
//+
Line(73) = {36, 42};
//+
Line(74) = {42, 44};
//+
Line(75) = {44, 43};
//+
Line(76) = {43, 41};
//+
Line(77) = {41, 42};
//+
Line(78) = {35, 41};
//+
Line(79) = {8, 44};
//+
Line(80) = {5, 43};
//+
Line(81) = {43, 17};
//+
Line(82) = {44, 19};
//+
Line(83) = {8, 23};
//+
Line(84) = {5, 21};
//+
Line(85) = {35, 22};
//+
Line(86) = {36, 24};
//+
Line(87) = {42, 20};
//+
Line(88) = {41, 18};
//+
Line(89) = {39, 35};
//+
Line(90) = {40, 36};
//+
Line(91) = {48, 42};
//+
Line(92) = {47, 41};
//+
Line(93) = {46, 3};
//+
Line(94) = {45, 2};
//+
Line(95) = {47, 37};
//+
Line(96) = {48, 38};
//+
Line(97) = {41, 33};
//+
Line(98) = {43, 1};
//+
Line(99) = {44, 4};
//+
Line(100) = {42, 34};
//+
Line(101) = {34, 38};
//+
Line(102) = {33, 37};
//+
Line(103) = {20, 12};
//+
Line(104) = {18, 10};
//+
Line(105) = {19, 11};
//+
Line(106) = {17, 9};
//+
Line(107) = {25, 18};
//+
Line(108) = {31, 24};
//+
Line(109) = {27, 20};
//+
Line(110) = {29, 22};
//+
Line(111) = {28, 16};
//+
Line(112) = {26, 14};
//+
Line(113) = {25, 13};
//+
Line(114) = {27, 15};
//+
Transfinite Curve {23, 17, 18, 24, 82, 81, 84, 85, 86, 83, 87, 88, 10, 9, 12, 11, 49, 55, 50, 51, 52, 53, 54, 56} = nrd+1 Using Progression 1/g;
//+
Transfinite Curve {1, 4, 3, 2, 8, 5, 7, 6, 40, 37, 38, 39, 28, 25, 26, 27, 36, 34, 33, 29, 32, 31, 30, 41, 44, 42, 43, 69, 78, 73, 77, 74, 75, 76, 71, 72, 70, 79, 80, 65, 60, 59, 58, 67, 68, 62, 57, 64, 63, 61, 16, 13, 19, 14, 15, 20, 21, 22, 48, 47, 45, 46, 35, 66} = nrf+1 Using Progression 1;
//+
Transfinite Curve {93, 96, 94, 95, 100, 97, 99, 98, 111, 112, 114, 113, 103, 104, 105, 106} = nrv+1 Using Progression 1;
//+
Transfinite Curve {89, 90, 110, 108, 107, 109, 92, 91, 101, 102} = nrl+1 Using Progression 1;

// ---- Surfaces and volumes (26 hex blocks)
// Down-leg (X1)
Curve Loop(1) = {22, 19, 20, 21}; Surface(1) = {1}; Transfinite Surface{1} = {9, 10, 12, 11};
Curve Loop(2) = {-58, -57, -60, -59}; Surface(2) = {2}; Transfinite Surface{2} = {17, 18, 20, 19};
Curve Loop(3) = {22, -104, 58, 106}; Surface(3) = {3}; Transfinite Surface{3} = {9, 10, 18, 17};
Curve Loop(4) = {19, -103, 57, 104}; Surface(4) = {4}; Transfinite Surface{4} = {10, 12, 20, 18};
Curve Loop(5) = {20, -105, 60, 103}; Surface(5) = {5}; Transfinite Surface{5} = {12, 11, 19, 20};
Curve Loop(6) = {21, -106, 59, 105}; Surface(6) = {6}; Transfinite Surface{6} = {11, 9, 17, 19};
Surface Loop(1) = {1, 2, 3, 4, 5, 6}; Volume(1) = {1}; Transfinite Volume{1} = {9, 10, 12, 11, 17, 18, 20, 19};  // tube
Curve Loop(7) = {-14, 18, -22, -24}; Surface(7) = {7}; Transfinite Surface{7} = {1, 33, 10, 9};
Curve Loop(8) = {76, 88, 58, -81}; Surface(8) = {8}; Transfinite Surface{8} = {43, 41, 18, 17};
Curve Loop(9) = {-14, -97, -76, 98}; Surface(9) = {9}; Transfinite Surface{9} = {1, 33, 41, 43};
Curve Loop(10) = {18, -104, -88, 97}; Surface(10) = {10}; Transfinite Surface{10} = {33, 10, 18, 41};
Curve Loop(11) = {-24, -98, 81, 106}; Surface(11) = {11}; Transfinite Surface{11} = {9, 1, 43, 17};
Surface Loop(2) = {7, 8, 9, 10, 3, 11}; Volume(2) = {2}; Transfinite Volume{2} = {1, 33, 10, 9, 43, 41, 18, 17};  // -y
Curve Loop(12) = {-13, 17, -19, -18}; Surface(12) = {12}; Transfinite Surface{12} = {33, 34, 12, 10};
Curve Loop(13) = {77, 87, 57, -88}; Surface(13) = {13}; Transfinite Surface{13} = {41, 42, 20, 18};
Curve Loop(14) = {-13, -100, -77, 97}; Surface(14) = {14}; Transfinite Surface{14} = {33, 34, 42, 41};
Curve Loop(15) = {17, -103, -87, 100}; Surface(15) = {15}; Transfinite Surface{15} = {34, 12, 20, 42};
Surface Loop(3) = {12, 13, 14, 15, 4, 10}; Volume(3) = {3}; Transfinite Volume{3} = {33, 34, 12, 10, 41, 42, 20, 18};  // +x
Curve Loop(16) = {-16, 23, -20, -17}; Surface(16) = {16}; Transfinite Surface{16} = {34, 4, 11, 12};
Curve Loop(17) = {74, 82, 60, -87}; Surface(17) = {17}; Transfinite Surface{17} = {42, 44, 19, 20};
Curve Loop(18) = {-16, -99, -74, 100}; Surface(18) = {18}; Transfinite Surface{18} = {34, 4, 44, 42};
Curve Loop(19) = {23, -105, -82, 99}; Surface(19) = {19}; Transfinite Surface{19} = {4, 11, 19, 44};
Surface Loop(4) = {16, 17, 18, 19, 5, 15}; Volume(4) = {4}; Transfinite Volume{4} = {34, 4, 11, 12, 42, 44, 19, 20};  // +y
Curve Loop(20) = {-15, 24, -21, -23}; Surface(20) = {20}; Transfinite Surface{20} = {4, 1, 9, 11};
Curve Loop(21) = {75, 81, 59, -82}; Surface(21) = {21}; Transfinite Surface{21} = {44, 43, 17, 19};
Curve Loop(22) = {-15, -98, -75, 99}; Surface(22) = {22}; Transfinite Surface{22} = {4, 1, 43, 44};
Surface Loop(5) = {20, 21, 22, 11, 6, 19}; Volume(5) = {5}; Transfinite Volume{5} = {4, 1, 9, 11, 44, 43, 17, 19};  // -x

// Corner 1
Curve Loop(23) = {-62, -61, -64, -63}; Surface(23) = {23}; Transfinite Surface{23} = {21, 22, 24, 23};
Curve Loop(24) = {-58, 66, 62, 67}; Surface(24) = {24}; Transfinite Surface{24} = {17, 18, 22, 21};
Curve Loop(25) = {-57, -65, 61, -66}; Surface(25) = {25}; Transfinite Surface{25} = {18, 20, 24, 22};
Curve Loop(26) = {-60, -68, 64, 65}; Surface(26) = {26}; Transfinite Surface{26} = {20, 19, 23, 24};
Curve Loop(27) = {-59, -67, 63, 68}; Surface(27) = {27}; Transfinite Surface{27} = {19, 17, 21, 23};
Surface Loop(6) = {2, 23, 24, 25, 26, 27}; Volume(6) = {6}; Transfinite Volume{6} = {17, 18, 20, 19, 21, 22, 24, 23};  // tube
Curve Loop(28) = {76, -78, 70, 80}; Surface(28) = {28}; Transfinite Surface{28} = {43, 41, 35, 5};
Curve Loop(29) = {66, -85, 78, 88}; Surface(29) = {29}; Transfinite Surface{29} = {18, 22, 35, 41};
Curve Loop(30) = {62, -84, -70, 85}; Surface(30) = {30}; Transfinite Surface{30} = {22, 21, 5, 35};
Curve Loop(31) = {67, -81, -80, 84}; Surface(31) = {31}; Transfinite Surface{31} = {21, 17, 43, 5};
Surface Loop(7) = {24, 28, 8, 29, 30, 31}; Volume(7) = {7}; Transfinite Volume{7} = {17, 18, 22, 21, 43, 41, 35, 5};  // -y
Curve Loop(32) = {-74, -73, -72, 79}; Surface(32) = {32}; Transfinite Surface{32} = {44, 42, 36, 8};
Curve Loop(33) = {-65, -86, 73, 87}; Surface(33) = {33}; Transfinite Surface{33} = {20, 24, 36, 42};
Curve Loop(34) = {-64, -83, 72, 86}; Surface(34) = {34}; Transfinite Surface{34} = {24, 23, 8, 36};
Curve Loop(35) = {68, -82, -79, 83}; Surface(35) = {35}; Transfinite Surface{35} = {23, 19, 44, 8};
Surface Loop(8) = {26, 32, 17, 33, 34, 35}; Volume(8) = {8}; Transfinite Volume{8} = {19, 20, 24, 23, 44, 42, 36, 8};  // +y
Curve Loop(36) = {-75, -79, -71, 80}; Surface(36) = {36}; Transfinite Surface{36} = {43, 44, 8, 5};
Curve Loop(37) = {-63, -84, 71, 83}; Surface(37) = {37}; Transfinite Surface{37} = {23, 21, 5, 8};
Surface Loop(9) = {27, 36, 21, 35, 37, 31}; Volume(9) = {9}; Transfinite Volume{9} = {17, 19, 23, 21, 43, 44, 8, 5};  // -x
Curve Loop(38) = {-70, -69, -72, -71}; Surface(38) = {38}; Transfinite Surface{38} = {5, 35, 36, 8};
Curve Loop(39) = {-61, -86, 69, 85}; Surface(39) = {39}; Transfinite Surface{39} = {22, 24, 36, 35};
Surface Loop(10) = {23, 38, 30, 39, 34, 37}; Volume(10) = {10}; Transfinite Volume{10} = {21, 22, 24, 23, 5, 35, 36, 8};  // +z

// Horizontal leg
Curve Loop(40) = {27, -34, -31, 36}; Surface(40) = {40}; Transfinite Surface{40} = {25, 27, 31, 29};
Curve Loop(41) = {-57, -109, -27, 107}; Surface(41) = {41}; Transfinite Surface{41} = {18, 20, 27, 25};
Curve Loop(42) = {-65, -108, 34, 109}; Surface(42) = {42}; Transfinite Surface{42} = {20, 24, 31, 27};
Curve Loop(43) = {61, -110, 31, 108}; Surface(43) = {43}; Transfinite Surface{43} = {24, 22, 29, 31};
Curve Loop(44) = {-66, -107, -36, 110}; Surface(44) = {44}; Transfinite Surface{44} = {22, 18, 25, 29};
Surface Loop(11) = {25, 40, 41, 42, 43, 44}; Volume(11) = {11}; Transfinite Volume{11} = {18, 20, 24, 22, 25, 27, 31, 29};  // tube
Curve Loop(45) = {-36, -52, 47, 54}; Surface(45) = {45}; Transfinite Surface{45} = {25, 29, 39, 47};
Curve Loop(46) = {-85, -89, 52, 110}; Surface(46) = {46}; Transfinite Surface{46} = {22, 35, 39, 29};
Curve Loop(47) = {78, -92, -47, 89}; Surface(47) = {47}; Transfinite Surface{47} = {35, 41, 47, 39};
Curve Loop(48) = {88, -107, -54, 92}; Surface(48) = {48}; Transfinite Surface{48} = {41, 18, 25, 47};
Surface Loop(12) = {29, 45, 44, 46, 47, 48}; Volume(12) = {12}; Transfinite Volume{12} = {18, 22, 35, 41, 25, 29, 39, 47};  // -y
Curve Loop(49) = {-54, 39, 56, -27}; Surface(49) = {49}; Transfinite Surface{49} = {25, 47, 48, 27};
Curve Loop(50) = {77, -91, -39, 92}; Surface(50) = {50}; Transfinite Surface{50} = {41, 42, 48, 47};
Curve Loop(51) = {87, -109, -56, 91}; Surface(51) = {51}; Transfinite Surface{51} = {42, 20, 27, 48};
Surface Loop(13) = {13, 49, 48, 50, 51, 41}; Volume(13) = {13}; Transfinite Volume{13} = {18, 41, 42, 20, 25, 47, 48, 27};  // -z
Curve Loop(52) = {-56, -48, 53, 34}; Surface(52) = {52}; Transfinite Surface{52} = {27, 48, 40, 31};
Curve Loop(53) = {-73, -90, 48, 91}; Surface(53) = {53}; Transfinite Surface{53} = {42, 36, 40, 48};
Curve Loop(54) = {86, -108, -53, 90}; Surface(54) = {54}; Transfinite Surface{54} = {36, 24, 31, 40};
Surface Loop(14) = {33, 52, 51, 53, 54, 42}; Volume(14) = {14}; Transfinite Volume{14} = {20, 42, 36, 24, 27, 48, 40, 31};  // +y
Curve Loop(55) = {31, -53, -43, 52}; Surface(55) = {55}; Transfinite Surface{55} = {29, 31, 40, 39};
Curve Loop(56) = {69, -89, 43, 90}; Surface(56) = {56}; Transfinite Surface{56} = {36, 35, 39, 40};
Surface Loop(15) = {39, 55, 43, 54, 56, 46}; Volume(15) = {15}; Transfinite Volume{15} = {22, 24, 36, 35, 29, 31, 40, 39};  // +z

// Corner 2
Curve Loop(57) = {-26, -25, -28, -27}; Surface(57) = {57}; Transfinite Surface{57} = {25, 26, 28, 27};
Curve Loop(58) = {-30, -29, -32, -31}; Surface(58) = {58}; Transfinite Surface{58} = {29, 30, 32, 31};
Curve Loop(59) = {-26, -35, 30, 36}; Surface(59) = {59}; Transfinite Surface{59} = {25, 26, 30, 29};
Curve Loop(60) = {-25, -33, 29, 35}; Surface(60) = {60}; Transfinite Surface{60} = {26, 28, 32, 30};
Curve Loop(61) = {-28, -34, 32, 33}; Surface(61) = {61}; Transfinite Surface{61} = {28, 27, 31, 32};
Surface Loop(16) = {57, 58, 59, 60, 61, 40}; Volume(16) = {16}; Transfinite Volume{16} = {25, 26, 28, 27, 29, 30, 32, 31};  // tube
Curve Loop(62) = {-38, -46, 42, 47}; Surface(62) = {62}; Transfinite Surface{62} = {47, 45, 6, 39};
Curve Loop(63) = {-26, -55, 38, 54}; Surface(63) = {63}; Transfinite Surface{63} = {25, 26, 45, 47};
Curve Loop(64) = {-35, -51, 46, 55}; Surface(64) = {64}; Transfinite Surface{64} = {26, 30, 6, 45};
Curve Loop(65) = {30, -52, -42, 51}; Surface(65) = {65}; Transfinite Surface{65} = {30, 29, 39, 6};
Surface Loop(17) = {59, 62, 63, 64, 65, 45}; Volume(17) = {17}; Transfinite Volume{17} = {25, 26, 30, 29, 47, 45, 6, 39};  // -y
Curve Loop(66) = {40, -45, -44, 48}; Surface(66) = {66}; Transfinite Surface{66} = {48, 46, 7, 40};
Curve Loop(67) = {28, -49, -40, 56}; Surface(67) = {67}; Transfinite Surface{67} = {27, 28, 46, 48};
Curve Loop(68) = {-33, -50, 45, 49}; Surface(68) = {68}; Transfinite Surface{68} = {28, 32, 7, 46};
Curve Loop(69) = {-32, -53, 44, 50}; Surface(69) = {69}; Transfinite Surface{69} = {32, 31, 40, 7};
Surface Loop(18) = {61, 66, 67, 68, 69, 52}; Volume(18) = {18}; Transfinite Volume{18} = {27, 28, 32, 31, 48, 46, 7, 40};  // +y
Curve Loop(70) = {-37, -45, 41, 46}; Surface(70) = {70}; Transfinite Surface{70} = {45, 46, 7, 6};
Curve Loop(71) = {-25, -49, 37, 55}; Surface(71) = {71}; Transfinite Surface{71} = {26, 28, 46, 45};
Curve Loop(72) = {29, -51, -41, 50}; Surface(72) = {72}; Transfinite Surface{72} = {32, 30, 6, 7};
Surface Loop(19) = {60, 70, 71, 68, 72, 64}; Volume(19) = {19}; Transfinite Volume{19} = {26, 28, 32, 30, 45, 46, 7, 6};  // +x
Curve Loop(73) = {-42, -41, -44, -43}; Surface(73) = {73}; Transfinite Surface{73} = {39, 6, 7, 40};
Surface Loop(20) = {58, 73, 65, 72, 69, 55}; Volume(20) = {20}; Transfinite Volume{20} = {29, 30, 32, 31, 39, 6, 7, 40};  // +z

// Up-leg (X2)
Curve Loop(74) = {-6, -5, -8, -7}; Surface(74) = {74}; Transfinite Surface{74} = {13, 14, 16, 15};
Curve Loop(75) = {-6, -112, 26, 113}; Surface(75) = {75}; Transfinite Surface{75} = {13, 14, 26, 25};
Curve Loop(76) = {-5, -111, 25, 112}; Surface(76) = {76}; Transfinite Surface{76} = {14, 16, 28, 26};
Curve Loop(77) = {-8, -114, 28, 111}; Surface(77) = {77}; Transfinite Surface{77} = {16, 15, 27, 28};
Curve Loop(78) = {-7, -113, 27, 114}; Surface(78) = {78}; Transfinite Surface{78} = {15, 13, 25, 27};
Surface Loop(21) = {74, 57, 75, 76, 77, 78}; Volume(21) = {21}; Transfinite Volume{21} = {13, 14, 16, 15, 25, 26, 28, 27};  // tube
Curve Loop(79) = {-2, 11, 6, -10}; Surface(79) = {79}; Transfinite Surface{79} = {37, 2, 14, 13};
Curve Loop(80) = {-2, -94, 38, 95}; Surface(80) = {80}; Transfinite Surface{80} = {37, 2, 45, 47};
Curve Loop(81) = {11, -112, -55, 94}; Surface(81) = {81}; Transfinite Surface{81} = {2, 14, 26, 45};
Curve Loop(82) = {-10, -95, 54, 113}; Surface(82) = {82}; Transfinite Surface{82} = {13, 37, 47, 25};
Surface Loop(22) = {79, 63, 80, 81, 75, 82}; Volume(22) = {22}; Transfinite Volume{22} = {37, 2, 14, 13, 47, 45, 26, 25};  // -y
Curve Loop(83) = {-1, 12, 5, -11}; Surface(83) = {83}; Transfinite Surface{83} = {2, 3, 16, 14};
Curve Loop(84) = {-1, -93, 37, 94}; Surface(84) = {84}; Transfinite Surface{84} = {2, 3, 46, 45};
Curve Loop(85) = {12, -111, -49, 93}; Surface(85) = {85}; Transfinite Surface{85} = {3, 16, 28, 46};
Surface Loop(23) = {83, 71, 84, 85, 76, 81}; Volume(23) = {23}; Transfinite Volume{23} = {2, 3, 16, 14, 45, 46, 28, 26};  // +x
Curve Loop(86) = {-4, 9, 8, -12}; Surface(86) = {86}; Transfinite Surface{86} = {3, 38, 15, 16};
Curve Loop(87) = {-4, -96, 40, 93}; Surface(87) = {87}; Transfinite Surface{87} = {3, 38, 48, 46};
Curve Loop(88) = {9, -114, -56, 96}; Surface(88) = {88}; Transfinite Surface{88} = {38, 15, 27, 48};
Surface Loop(24) = {86, 67, 87, 88, 77, 85}; Volume(24) = {24}; Transfinite Volume{24} = {3, 38, 15, 16, 46, 48, 27, 28};  // +y
Curve Loop(89) = {-3, 10, 7, -9}; Surface(89) = {89}; Transfinite Surface{89} = {38, 37, 13, 15};
Curve Loop(90) = {-3, -95, 39, 96}; Surface(90) = {90}; Transfinite Surface{90} = {38, 37, 47, 48};
Surface Loop(25) = {89, 49, 90, 82, 78, 88}; Volume(25) = {25}; Transfinite Volume{25} = {38, 37, 13, 15, 48, 47, 25, 27};  // -x

// Between the legs
Curve Loop(91) = {102, 3, -101, 13}; Surface(91) = {91}; Transfinite Surface{91} = {33, 37, 38, 34};
Curve Loop(92) = {102, -95, 92, 97}; Surface(92) = {92}; Transfinite Surface{92} = {33, 37, 47, 41};
Curve Loop(93) = {-101, -100, -91, 96}; Surface(93) = {93}; Transfinite Surface{93} = {38, 34, 42, 48};
Surface Loop(26) = {91, 50, 92, 90, 93, 14}; Volume(26) = {26}; Transfinite Volume{26} = {33, 37, 38, 34, 41, 47, 48, 42};  // block

Recombine Surface {1:93};

Physical Volume("lateral")   = {1, 6, 11, 16, 21};
Physical Volume("reservoir") = {2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26};
//+
Recursive Delete {
  Surface{93};
}
//+
Recursive Delete {
  Surface{93};
}
//+
Recursive Delete {
  Volume{26};
}
