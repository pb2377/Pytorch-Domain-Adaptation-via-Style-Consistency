# Pytorch-Domain-Adaptation-via-Style-Consistency

My PyTorch implementation of Domain Adaptation for Object Detection via Style Consistency (https://arxiv.org/abs/1911.10033) developed from base SSD pytorch implementation (https://github.com/amdegroot/ssd.pytorch) and Ada In fast styl transfer (https://github.com/amdegroot/ssd.pytorch).


Results:
Baseline Models - Recognition performance given the base SSD model pretrained on Pascal VOC from the paper and this pytorch implementation (i.e. the pretrained weights from amdegroot/ssd.pytorch)
Main Models - Performance of their full model from the paper and this pytorch implementation (ST indicates sytle transfer consistency and RPL indicated robust pseudolabelling)

My implementation produces near-identical mAP to the original implementation's performance listed in the paper (class-specific AP varies a little, but not significantly). The only functional difference between this pytorch version and the original is that the paper implies they use a single style source to stylise each batch of training examples, which implies online style transfer training data. I run stylisation as preprocessing step to get N stylised copies of each annotated photograph, and then sample from these randomly from these -- i.e. here a style image stylises a batch worth of training images, but these stylised images are sampled separately.
#### TBD:
* Implement with Faster RCNN
  
 
### Pascal VOC -> Clipart:
|   | aero | bike | bird | boat | bttle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 19.8 | 49.5 | 20.1 | 23.0 | 11.3 | 38.6 | 34.2 | 2.5 | 39.1 | 21.6 | 27.3 | 10.8 | 32.5 | 54.1 | 45.3 | 31.2 | 19.0 | 19.5 | 19.1 | 17.9 | 26.8 |
| Pytorch | 20.4 | 55.2 | 20.5 | 25.2 | 17.2 | 33.5 | 32.9| 8.2| 44.6 | 11.0 | 30.1 | 7.8 | 26.5 | 47.0 | 36.7 | 28.3 | 2.3 | 20.2 | 26.6 | 19.8 | 25.7 |
| Main Models|
| Paper ST | 35.0 | 57.3 | 24.7 | 41.9 | 28.0 | 56.8 | 49.1 | 9.9 | 49.3 | 55.6 | 44.0 | 16.5 | 42.3 | 83.1 | 65.0 | 42.8 | 17.7 | 43.9 | 42.0 | 52.6 | 42.9 |
| Paper ST+RPL | 36.9 | 55.1 | 26.4 | 42.7 | 23.6 | 64.4 | 52.1 | 10.1 | 50.9 | 57.2 | 48.2 | 16.2 | 45.9 | 83.7 | 69.5 | 41.5 | 21.6 | 46.1 | 48.3 | 55.7 | 44.8 |
| Pytorch ST | 23.5 | 63.2 | 25.5 | 45.6 | 35.2 | 60.6 | 52.1 | 8.2 | 52.1 | 56.5 | 53.4 | 14.7 | 35.0 | 65.4 | 66.3 | 48.4 | 20.8 | 33.3 | 44.1 | 52.7| 42.8 |
| Pytorch ST+RPL | 25.3 | 68.8 | 25.8 | 43.8 | 30.9 | 53.9 | 58.6 | 9.0 | 53.2 | 61.4 | 56.3 | 17.2 | 39.9| 76.3 | 71.6 | 49.2 | 25.1 | 32.9 | 47.5 | 55.2 | 45.1 |



### Pascal VOC -> Watercolor:
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 79.8 | 49.5 | 38.1 | 35.1 | 30.4 | 65.1 | 49.6 |
| Pytorch | 77.5 | 48.5 | 44.6 | 30.3 | 27.9 | 62.7 | 48.6 | 
| Main Models|
| Paper ST | 81.4| 54.3 | 47.5 | 40.5 | 35.7 | 68.3 | 54.6 |
| Paper ST+RPL | 79.9 | 56.5 | 48.6 | 42.1 | 42.9 | 73.7 | 57.3 |
| Pytorch ST | 100.0 | 51.0 | 54.5 | 36.3 | 39.4 | 70.9 | 58.7 | 
| Pytorch ST+RPL | 98.2 | 54.3 | 56.0 | 41.1 | 43.5 | 76.0 | 61.5 | 


### Pascal VOC -> Comic:
|   | bike | bird | car | cat | dog | person | mAP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Models |
| Paper | 43.9 | 10.0 | 19.4 | 12.9 | 20.3 | 42.6 | 24.9 | 
| Pytorch | 45.4 | 9.7 | 25.6 | 9.8 | 10.9 | 36.7 | 23.0 | 
| Main Models|
| Paper ST | 51.4 | 17.3 | 39.9 | 21.4 | 31.9 | 56.1 | 36.3 |
| Paper ST+RPL | 55.9 | 19.7 | 42.3 | 23.6 | 31.5 | 63.4 | 39.4 |
| Pytorch ST | 49.5 | 14.9 | 38.4 | 24.3 | 28.7 | 60.6 | 36.1 | 
| Pytorch ST+RPL | 55.5 | 15.7 | 39.7 | 30.6 | 28.6 | 66.4 | 39.4 | 
