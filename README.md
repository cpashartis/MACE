# MACE - Mesh Adaptation through Configurational Energy

<img src="https://github.com/cpashartis/MACE/assets/7492783/ad964431-191d-44d3-99dd-ed4db25fccd1" width="400">

MACE is a sophisticated algorithm designed to optimize and reconstruct surfaces from point clouds by leveraging the principles of configurational energy. This project encapsulates the core of what makes dynamic mesh adaptation possible, providing users with the tools to transform chaotic sets of points into structured, optimized meshes. Ideal for applications in computational geometry, graphics, and scientific visualization, MACE bridges the gap between raw data and actionable geometry. It was originally developped as a way to get a set of points defining a surface when given a set of points defining a 'thick' surface which would throw off any triangulation technique.

As it stands, it could use some optimization, but it is a proof of concept and demonstrates how simulated annealing can be used for such a purpose. Perhaps in the future adding a penalty to the objective function for points being too close is a good idea :)

## Features

- **Point Cloud to Mesh Conversion**: Convert unstructured point clouds into structured mesh representations.
- **Energy-Based Optimization**: Utilize configurational energy principles to optimize the placement and connectivity of mesh points.
- **Adaptive Meshing**: Dynamically adjust mesh density and structure based on local geometric features.
- **Interactive Visualization**: Explore the evolution of mesh adaptation through an interactive interface.

## Installation

To get started with MACE, clone this repository, setup your environment, then try ```python setup.py install``` or ```python setup.py develop```

## Running MACE
### Load your point cloud (example)
```
import numpy as np
points = np.random.rand(100, 3)  # Example point cloud data, which we want to fit a surface to

# Initialize MACE
import mace
mace_inst = MeshAdaptation(points)

# Run the optimization
# MACE by default contracts points on the boundary of a box to the generate a surface on the set of points you gave it.
optimized_mesh = mace_inst.simulated_annealing(points) 

# Loading MACE data
mace.load_configurations(file_path)
```
## Example of the Contraction or Vacuuming Forming:
This code can be found in the notebook in the examples directory

### Before (box initial starting points)
<img src="https://github.com/cpashartis/MACE/assets/7492783/72a6aa77-684c-4ac8-a7c3-b3771290600d" width="400">

### After 28 Iterations
We begin to converge to the set of 'thick' points defined by the data. It now looks more like a fermi surface :)

<img src="https://github.com/cpashartis/MACE/assets/7492783/4d645166-f497-409d-a805-7fce1cb3f769" width="400">


# Advanced Usage
For advanced configurations and usage scenarios, please refer to the documentation or the examples directory in this repository. Including the widgets given above.
