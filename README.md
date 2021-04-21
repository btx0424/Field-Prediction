# Field-Prediction

### Data Preparation:

​	Run "convert_to_np" in **utils.data** with specified source and target directories to generate data in *.npy* format. By default, all fields are retained. This would produce output of shape (H, W, I, C) where H, W and I describes the dimensions of the mesh (I, which stands for *z-axis*, is actually not used here), and C is the number of fields.

### Training:

​	Run **train.py** with config files. Remember to set the **gpus** parameter for the trainer (add a python **argparser** if needed). Refer to the documentation for more details.