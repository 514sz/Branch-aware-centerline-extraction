# Branch-aware-centerline-extraction

Code for our paper "[Branch-Aware Double DQN for Centerline Extraction in Coronary CT Angiography](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_4)". 

- Propose the Double Deep Q-network based coronary artery tracing method in CCTA for the first time.
- Extracts the entire coronary tree with lower time-cost than other state-of-the-art methods, uses only one seed and terminates tracing automatically.


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
