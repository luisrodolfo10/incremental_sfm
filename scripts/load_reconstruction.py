import numpy as np
import os
import cv2
from sfm.gui import SimpleReconstrictionViewer, ReconstructionViewer
from open3d.visualization import gui


if __name__ == "__main__":
    dataset = "dog"

    image_dir = "data/images/"+ dataset
    recons_dir = "data/reconstructions/" + dataset + "/"

    pcd_file = recons_dir + "dog_fine.ply"
    cameras_file = recons_dir + "dog_fine-cams.npz"

    imgs_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    imgs_paths = [os.path.join(image_dir, f) for f in imgs_list]

    img = cv2.imread(imgs_paths[0])
    img_size = (img.shape[1], img.shape[0])
    
    #viewer = SimpleReconstrictionViewer(pcd_file, cameras_file, img_size=img_size)
    #gui.Application.instance.initialize()
    viewer = ReconstructionViewer(pcd_file, cameras_file, img_size=img_size, imgs_paths=imgs_paths)
    viewer.run()
    #gui.Application.instance.run()