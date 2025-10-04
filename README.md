## General Description 📝
This repo contains training code and training steps for finetuning of Masked R-CNN to the Cityscapes dataset. Cityscapes is a dataset that consists of urban scenes from diverse German cities and provides a good benchmark for instance segmentation/object detection models. 

## Dataset Description 🗂️
We use the Cityscapes dataset [4], a large-scale dataset
 that contains diverse urban scenes from various German
 cities. It consists of more than 3000 high-quality images
 (2048px by 1024px) with instances of objects that are com
monly encountered during driving (pedestrians, cars etc.)
 and their respective instance annotations and pixel-level in
stance segmentation masks. There are also more than 20000
 semantically annotated images that can be used for seman
tic annotation and pre-training tasks.
 For our experiments, 2,975 finely annotated images are
 used for training and 500 for validation. Of the dataset,
 1,525 images are reserved for benchmarking and lack pub
lic labels, so they were excluded from training. The coarse
 annotations from the trainextra split were utilized to
 provide additional semantically annotated samples that we
 used to pretrain resnet50 backbone that we used in our Mask
 R-CNN implementation. Due to storage and memory con
straints, we used only 4120 coarsely annotated images.

## The Training Procedure ⚙️
Explanation of the model architecture, hyperparameters, and training steps.

## Overcomed Issues 🛠️
Challenges faced during training and how they were resolved.

## The Final Test Set Metrics 📊
Evaluation results, metrics (mIoU, accuracy), and visualizations.

## Future Projects 🚀
Planned improvements, extensions, or experiments for this project.



## Inference Examples
![Alt text](inference_examples/inf1.png)
![Alt text](inference_examples/inf2.png)
![Alt text](inference_examples/inf3.png)







