#include "format.hpp"
#include <cstddef>
complexFormat** zernikeDecomposition(complexFormat* img, int maxn, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL);
complexFormat** laguerreDecomposition(complexFormat* img, int maxn, int maxl, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL);
