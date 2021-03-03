# Pytorch-Domain-Adaptation-via-Style-Consistency

My PyTorch implementation of Domain Adaptation for Object Detection via Style Consistency (https://arxiv.org/abs/1911.10033) developed from base SSD pytorch implementation (https://github.com/amdegroot/ssd.pytorch) and Ada In fast styl transfer (https://github.com/amdegroot/ssd.pytorch).


Results:
Baseline Models - Recognition performance given the base SSD model pretrained on Pascal VOC from the paper and this pytorch implementation (i.e. the pretrained weights from amdegroot/ssd.pytorch)
Main Models - Performance of their full model (robust pseudolabelling + style consistency) from the paper and this pytorch implementation


  

#  
Pascal VOC -> Clipart:
|   | aero | bike | bird | boat | bttle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Main Models|
| Paper | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44.9 |


#
Pascal VOC -> Watercolor
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Main Models|
| Paper| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 


#
Pascal VOC -> Comic
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Main Models|
| Paper| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
