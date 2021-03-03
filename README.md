# Pytorch-Domain-Adaptation-via-Style-Consistency

My PyTorch implementation of Domain Adaptation for Object Detection via Style Consistency (https://arxiv.org/abs/1911.10033) developed from base SSD pytorch implementation (https://github.com/amdegroot/ssd.pytorch) and Ada In fast styl transfer (https://github.com/amdegroot/ssd.pytorch).


Results:
Baseline Models - Recognition performance given the base SSD model pretrained on Pascal VOC from the paper and this pytorch implementation (i.e. the pretrained weights from amdegroot/ssd.pytorch)
Main Models - Performance of their full model from the paper and this pytorch implementation (ST indicates sytle transfer consistency and RPL indicated robust pseudolabelling)

TBD
* add in full class breakdown of results for pytorch implementation (currently only listng my mAP)
* Implement with Faster RCNN
  

#  
Pascal VOC -> Clipart:
|   | aero | bike | bird | boat | bttle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 19.8 | 49.5 | 20.1 | 23.0 | 11.3 | 38.6 | 34.2 | 2.5 | 39.1 | 21.6 | 27.3 | 10.8 | 32.5 | 54.1 | 45.3 | 31.2 | 19.0 | 19.5 | 19.1 | 17.9 | 26.8 |
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Main Models|
| Paper ST | 35.0 | 57.3 | 24.7 | 41.9 | 28.0 | 56.8 | 49.1 | 9.9 | 49.3 | 55.6 | 44.0 | 16.5 | 42.3 | 83.1 | 65.0 | 42.8 | 17.7 | 43.9 | 42.0 | 52.6 | 42.9 |
| Paper ST+RPL | 36.9 | 55.1 | 26.4 | 42.7 | 23.6 | 64.4 | 52.1 | 10.1 | 50.9 | 57.2 | 48.2 | 16.2 | 45.9 | 83.7 | 69.5 | 41.5 | 21.6 | 46.1 | 48.3 | 55.7 | 44.8 |
| Pytorch ST | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 43.2 |
| Pytorch ST+RPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44.9 |


#
Pascal VOC -> Watercolor
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 79.8 | 49.5 | 38.1 | 35.1 | 30.4 | 65.1 | 49.6 |
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Main Models|
| Paper ST | 81.4| 54.3 | 47.5 | 40.5 | 35.7 | 68.3 | 54.6 |
| Paper ST+RPL |79.9 | 56.5 | 48.6 | 42.1 | 42.9 | 73.7 | 57.3 |
| Pytorch ST | 0 | 0 | 0 | 0 | 0 | 0 | 58.7 | 
| Pytorch ST+RPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 


#
Pascal VOC -> Comic
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 43.9 | 10.0 | 19.4 | 12.9 | 20.3 | 42.6 | 24.9 | 
| Pytorch | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Main Models|
| Paper ST | 51.4 | 17.3 | 39.9 | 21.4 | 31.9 | 56.1 | 36.3 |
| Paper ST+RPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch ST | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
| Pytorch ST+RPL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
