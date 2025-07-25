# Branch-aware-centerline-extraction

The PyTorch re-implement of a branch-aware coronary centerline extraction in CT Angiography images. (paper: '[Branch-Aware Double DQN for Centerline Extraction in Coronary CT Angiography](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_4)')

- Detecting the vessel branches automatically.
- Extracting the entire coronary tree with only one seed, and terminating the tracing process automatically.

## Key idea

A branch-aware coronary centerline extraction approach (BACCE) is introduced, which is based on Double Deep Q-Network (DDQN) and 3D dilated CNN. It consists of two parts: a DDQN based tracker and a branch-aware detector. The tracker predicts the next action of an agent to trace the centerline. The detector detects the bifurcation-points and radius of the coronary artery. The detector enable the BACCE to trace the coronary branches automatically. As a result, the BACCE only needs one seed at the coronary 'trunk' to extract the entire coronary tree.


<img src="https://github.com/514sz/Branch-aware-centerline-extraction/blob/master/images/framework.png" width="800" height="600">

A single seed is firsly placed at the coronary 'trunk'. The tracker starts from the seed. Meanwhile, the detector detects whether the tracker is located at a bifurcation-point or an endpoint, and estimates the vessel radius. At the bifurcation point, the ray-burst sampling algorithm is executed to detect branches, and the tracker will track the detected branches; at the endpoint, the tracker will terminate tracking the current branch, and continue to track other branches. This process is repeated until all branches have been extracted.

Two networks need to be trained:

- The tracker to track the coronary centerline.
- The detector to detect bifurcation-points and endpoints, and estimate the radius of the coronary artery.



## Installation

```
cd Branch-aware-centerline-extraction
pip install -r requirement.txt
```

## Training

### 1. Preparing CTA08 dataset

&#9888;
The website of CAT08 dataset is no longer accessible. You can contact Dr.Theo van Walsum (t.vanwalsum@erasmusmc.nl) to acquire this dataset.

1. Unzip training.tar.gz to:
```
    home/
        -zyy/
            -training/
                -dataset00/
                -dataset01/
                -dataset02/
                -dataset03/
                -dataset04/
                -dataset05/
                -dataset06/
                -dataset07/
```
2. Construct training data for the tracker:

```
cd Branch-aware-centerline-extraction
python3 Training_data_tracker.py
```

3. Construct training data for the detector:

```
cd Branch-aware-centerline-extraction
python3 Training_data_detector.py
```

training data are prepared:
```
-home/
    -zyy/
        -training/
            -dataset00/
                -vessel0/
                    vox_radi.npy
                    -patch/
                        patch0.npy
                        .
                        .
                    -detector_label/
                        label0.txt
                        .
                        . 
                .        
                .
            .
            .
```

    
### 2. Training models

1. Training tracker model
```
cd Branch-aware-centerline-extraction/
python3 Train_Tracker.py
```

2. Training detector model
```bash
cd Branch-aware-centerline-extraction/
python3 Train_Detector.py
```

### 3. Extracting coronary centerline tree

```
cd Branch-aware-centerline-extraction
python3 CenterlineTreeExtraction.py
```

The extracted coronary artery tree is as follows:

<img src="https://github.com/514sz/Image-store/blob/main/fig4.png" width="800" height="600">

## References

Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

    @inproceedings{zhang2020branch,
      title={Branch-aware double DQN for centerline extraction in coronary CT angiography},
      author={Zhang, Yuyang and Luo, Gongning and Wang, Wei and Wang, Kuanquan},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={35--44},
      year={2020},
      organization={Springer}
      }
