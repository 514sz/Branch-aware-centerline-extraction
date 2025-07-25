# Branch-aware-centerline-extraction

The PyTorch re-implement of a branch-aware coronary centerline extraction in CT Angiography images. (paper: '[Branch-Aware Double DQN for Centerline Extraction in Coronary CT Angiography](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_4)')

- Detecting the vessel branches automatically.
- Extracting the entire coronary tree with only one seed, and terminating the tracing process automatically.

## Key idea

A branch-aware coronary centerline extraction approach (BACCE) is introduced, which is based on Double Deep Q-Network (DDQN) and 3D dilated CNN. It consists of two parts: a DDQN based tracker and a branch-aware detector. The tracker predicts the next action of an agent to trace the centerline. The detector detects the bifurcation-points and radius of the coronary artery. The detector enable the BACCE to trace the coronary branches automatically. As a result, the BACCE only needs one seed at the coronary 'trunk' to extract the entire coronary tree.



## Requirements

Python 3.6.2

Pytorch 1.7

CUDA 11.2

## Coordinate transformation

```
python w_coor2v_coor.py
```
    
## Training

```
python ddqn.py
python detector.py

```

## Inference

```
python app.py
```

## Cite

Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.

    @inproceedings{zhang2020branch,
      title={Branch-aware double DQN for centerline extraction in coronary CT angiography},
      author={Zhang, Yuyang and Luo, Gongning and Wang, Wei and Wang, Kuanquan},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={35--44},
      year={2020},
      organization={Springer}
      }
