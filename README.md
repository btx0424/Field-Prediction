# Fluid Flow Field Prediction

Conclusion: directly using deep neural nets to predict the final result is not even close to a viable approach.

- A stationary (in time) target doesn't always exist. 
- The  numerical variation of the flow parameters (Mach, AoA, etc.) serves too weak as a conditioning factor. The model would degenerate to just output the average of what it have seen during training.
- Spatial discretization is crucial for high fidelity but is unfortunately tricky.
- Enforcing physical constraints (e.g. applying boundary conditions) may help but is difficult in formulation.
- [report]()

### Data Preparation: Converting Structured Block Mesh to Tensors

​	Run "convert_to_np" in **utils.data** with specified source and target directories to generate data in *.npy* format. By default, all fields are retained. This would produce output of shape (H, W, I, C) where H, W and I describes the dimensions of the mesh (I, which stands for *z-axis*, is actually not used here), and C is the number of fields.



### Model Architecture: An Image2Image Translation Analogy