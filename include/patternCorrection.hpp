#ifndef __PATTERNCORRECTION_H__
#define __PATTERNCORRECTION_H__
#include "format.hpp"

// Tilt correction (rotation around y-axis). Center always at (col/2, row/2).
// useSineLens: 0 = free-space (unified tilt+Ewald), 1 = sine lens (1D shear only)
// beamstop: Real** — if NULL, allocated (all 1s). OOB pixels set to 0 (free floating).
void reflCorrectToRegular(Real* tilted, Real* regular, Real** beamstop, Real tiltDeg, Real distance, Real pixelSize, int row, int column, int useSineLens);
void reflSimulateTilted(Real* regular, Real* tilted, Real tiltDeg, Real distance, Real pixelSize, int row, int column, int useSineLens);

// Ewald sphere curvature correction (no tilt). a = pixelSize / distance.
void ewaldCorrectToRegular(Real* detector, Real* regular, Real** beamstop, Real a, int row, int column, Real centerPx, Real centerPy);
void ewaldSimulateTilted(Real* regular, Real* detector, Real a, int row, int column, Real centerPx, Real centerPy);
#endif
