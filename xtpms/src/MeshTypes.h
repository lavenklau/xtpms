#pragma once

// OpenMesh headers are expected on the include path (e.g. vcpkg).
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

namespace xtpms {

template <class Traits = OpenMesh::DefaultTraits>
using TriMesh = OpenMesh::TriMesh_ArrayKernelT<Traits>;

using DefaultTriMesh = TriMesh<OpenMesh::DefaultTraits>;

} // namespace xtpms
