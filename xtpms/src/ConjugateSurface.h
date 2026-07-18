#pragma once

// Discrete conjugate surface construction from a minimal surface fundamental domain.
// Given a triangulated minimal surface patch with boundary, computes its conjugate
// surface (also a minimal surface) by integrating the cotangent-weighted edge vectors.

#include "MeshTypes.h"

#include <string>

namespace xtpms {

// Compute the conjugate surface of a minimal surface mesh.
// The input mesh should be a fundamental domain of a periodic minimal surface
// (has boundary). The result is written to outputFile in OBJ format.
// thetaDeg: Bonnet rotation angle in degrees. 90° = pure conjugate surface;
// 0° = original surface; intermediate values produce the associate family.
// Returns true on success.
// After computing, validates that the cotangent Laplacian at interior vertices
// vanishes (mean curvature H ≈ 0, a necessary condition for minimal surfaces).
bool computeConjugateSurface(DefaultTriMesh& mesh,
							 const std::string& outputFile,
							 double thetaDeg = 90.0);

} // namespace xtpms
