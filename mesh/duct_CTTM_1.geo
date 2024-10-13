/*********************************************************************
 *  From Gmsh tutorial 1 & 10
 *  Usage: gmsh -2 file.geo -o mesh.msh
 *********************************************************************/

radius   = 0.28;
x_inlet  = 0;
L_begin  = 1;
L_liner  = 0.4;
x_outlet = L_begin  + L_liner + L_begin;


// Typical triangle size
lc = 0.04;


/*
            7          6          5    
      8----------7----------6----------5
      |          |          |          |
    8 |          |9         |10        | 4
      |          |          |          |
      1----------2----------3----------4 
            1          2          3     
*/

eps=1e-6;

Point(1) = {x_inlet        ,    0, 0, lc};
Point(2) = {L_begin        ,    0, 0, lc};
Point(3) = {L_begin+L_liner,    0, 0, lc};
Point(4) = {x_outlet       ,    0, 0, lc};
Point(5) = {x_outlet       , radius, 0, lc};
Point(6) = {L_begin+L_liner, radius, 0, lc};
Point(7) = {L_begin        , radius, 0, lc};
Point(8) = {x_inlet        , radius, 0, lc};

Line(1) = {1,2}; 
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

L1 = L_begin - x_inlet;
L2 = L_liner;
L3 = x_outlet-L1-L2;
h  = radius; 

Printf("L1=%g",L1);
Printf("L2=%g",L2);
Printf("L3=%g",L3);

coef=1.;


// Definition of physical elements
inlet   = 1;
walls   = 2;
outlet  = 3;
liner_1   = 4;
liner_2   = 5;
fluid   = 60;

Physical Line(outlet)  = {4};
Physical Line(walls)   = {1,3,5,7};
Physical Line(liner_1) =   {2};
Physical Line(liner_2) =   {6};
Physical Line(inlet) =   {8};
  

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};

Plane Surface(1) = {1};

Physical Surface("fluid",fluid) = {1,2,3}; 
