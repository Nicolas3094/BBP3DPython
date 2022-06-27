## Container loading

 are 9 data files.

All of these files were contributed by M.S.W. Ratcliff (mspmax@swansea.ac.uk)

(i) Files thpack1,thpack2,...,thpack7

These files were generated and used in:

[1]  E.E. Bischoff and M.S.W. Ratcliff, "Issues in the development of 
     Approaches to Container Loading", OMEGA, vol.23, no.4, (1995) pp 377-390.

The procedure used to create these test problems is presented in the above 
paper.

These problems are single container loading problems, the objective being to 
maximise the volume utilisation of the container.

The format of these data files is:
Number of test problems (P)
For each problem p (p=1,...,P) the data has the format
shown in the following example:

     Example:

 60 2508405    the problem number p, seed number used in [1]
 587 233 220   container length, width, height
 10            number of box types n
 1  78 1 72 1 58 1 14
 2  107 1 57 1 57 1 11      where there is one line for each box type
 3 ...................
 etc for n lines
The line for each box type contains 8 numbers:                         
box type i, box length, 0/1 indicator
box width, 0/1 indicator
box height, 0/1 indicator
number of boxes of type i

After each box dimension the 0/1 indicates whether placement in the 
vertical orientation is permissible (=1) or not (=0)

This file contains the Bischoff/Ratcliff container loading problems as here but with the extended datasets also (eg, as used in Davies/Bischoff 1999).
He (re)generated all the datasets in the way described in the original paper (i.e. the first 7 are identical to the ones
already in the OR-library: thpack1-7) as he could not find sets 8-10 anywhere. The source code is available at http://www.cs.nott.ac.uk/~sda/research.shtml


http://people.brunel.ac.uk/~mastjjb/jeb/orlib/thpackinfo.html