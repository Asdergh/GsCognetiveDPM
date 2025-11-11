from pyvista import Plotter
import nibabel as nib


file = "/home/ram/Downloads/sub-0002_task-video_run-10_bold.nii.gz"
data = nib.load(file).get_fdata()
sample = data[..., 45]

pv = Plotter()
pv.add_volume(sample, opacity="sigmoid_9", cmap="coolwarm")
pv.set_background(color=[0.0, 0.0, 0.0])
pv.add_axes()
pv.show()
