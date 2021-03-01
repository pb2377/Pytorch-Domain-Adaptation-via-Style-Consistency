import os
import torch
import time
import eval
import copy
from utils.utils import export_images
from itertools import cycle
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def train(model, criterion, optimizer, train_loader, val_dataset, max_iter, output_path, log_freq=1000, test_freq=100,
          aux_criterion=None):
    iteration = 0
    epoch = 0
    accuracy_history = []
    best_map = 0.0
    best_model = copy.deepcopy(model.state_dict())

    # Tensorboard writer
    writer = SummaryWriter(log_dir=output_path)
    writer.add_scalar("Test_mAP", 0., 0)

    print("Training Classifier...")
    while iteration <= max_iter:
        for images, style_ims, targets in train_loader:
            optimizer.zero_grad()

            if iteration > max_iter:
                break

            if torch.cuda.is_available():
                style_ims = style_ims.cuda()
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]

            # combine image batch
            images = torch.cat((images, style_ims), 0)
            targets = targets * 2

            out, aux_outputs = model(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            loss_aux = 0.

            for idx in range(len(aux_outputs)):
                aux_ph = aux_outputs[idx][:images.size(0)//2, :, :, :]
                aux_st = aux_outputs[idx][images.size(0)//2:, :, :, :]
                loss_aux += aux_criterion(aux_ph, aux_st)

            loss = loss + loss_aux

            loss.backward()
            optimizer.step()

            if not iteration % log_freq:
                # make summary writer for loc lss, conf loss, consistency_loss,
                writer.add_scalar("Loss", loss.item(), iteration)
                writer.add_scalar("Conf_loss", loss_c.item(), iteration)
                writer.add_scalar("Loc_loss", loss_l.item(), iteration)
                writer.add_scalar("Style_loss", loss_aux, iteration)

            if iteration > 0 and not iteration % 5000:
                print('Saving state, iter:', iteration)
                save_path = os.path.join(output_path, 'weights', 'iteraton-{}.pth'.format(iteration))
                torch.save(model.state_dict(), save_path)

            if iteration >= max_iter - (10*test_freq):
                if iteration == max_iter or iteration % test_freq == 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    accuracy_history.append(eval.evaluate(model, val_dataset,
                                                          os.path.join(output_path, 'iteration-{}'.format(iteration))))
                    model.train()

                    # keep best  model
                    writer.add_scalar("Test_mAP", accuracy_history[-1]['mAP'], iteration)
                    if accuracy_history[-1]['mAP'] > best_map:
                        best_map = accuracy_history[-1]['mAP']
                        best_model = copy.deepcopy(model.state_dict())

                    # Add to summary writer
            else:
                if not iteration % 1000 and iteration > 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    acc = eval.evaluate(model, val_dataset, os.path.join(output_path, 'tempdets'.format(iteration)))
                    writer.add_scalar("Test_mAP", acc['mAP'], iteration)
                    model.train()

            iteration += 1
        epoch += 1

    return model, best_model, best_map, accuracy_history


def pseudolabel_train(model, ps_pair, sc_pair, optimizer, val_dataset, max_iter, output_path,
                      log_freq=1000, test_freq=100, aux_criterion=None):
    """
    As standard trainer but with mixed dataset of Pseudolabels and Labelled VOC data, and train much like "frustratingly
    easy few shot learning" and evaluate on batches of the source and novel (pseudo) examples.

    Also include the addition to only use the labelled examples to generate hard negatives from the synthesized artwork.
    :return:
    """
    ps_loader, criterion_ps = ps_pair  # pseudolabel dataloader and criterion
    sc_loader, criterion_sc = sc_pair  # souerce dataloader and criterion
    del ps_pair, sc_pair
    iteration = 0
    accuracy_history = []
    best_map = 0.0
    best_model = copy.deepcopy(model.state_dict())

    # Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_path, 'pseudolabel'))
    writer.add_scalar("Test_mAP", 0., 0)

    print("Training Classifier...")
    while iteration <= max_iter:
        for ps_batch, sc_batch in zip(cycle(ps_loader), sc_loader):
            ps_images, ps_style_ims, ps_targets = ps_batch
            sc_images, sc_style_ims, sc_targets = sc_batch

            optimizer.zero_grad()

            if iteration > max_iter:
                break

            # combine into single batch
            images_batch = torch.cat((ps_images, ps_style_ims, sc_images, sc_style_ims), 0)

            if torch.cuda.is_available():
                images_batch = images_batch.cuda()
                ps_targets = [ann.cuda() for ann in ps_targets]
                sc_targets = [ann.cuda() for ann in sc_targets]

            # Pass through model
            out, aux_outputs = model(images_batch)

            # Split outputs back to pseudolabel and source
            ps_out = []
            sc_out = []
            for output in out[:-1]:
                ps_out.append(output[:ps_images.size(0)*2])
                sc_out.append(output[ps_images.size(0)*2:])
            ps_out.append(out[-1])
            sc_out.append(out[-1])

            # Pseudolabel Example Losses
            ps_targets = ps_targets + ps_targets
            loss_l_ps, loss_c_ps = criterion_ps(ps_out, ps_targets)
            loss_ps = loss_l_ps + loss_c_ps
            loss_aux_ps = 0.
            # First half of batch in aux outputs are pseudolabel items
            sz_1 = ps_images.size(0)
            sz_2 = sz_1 * 2
            for idx in range(len(aux_outputs)):
                aux_base = aux_outputs[idx][:sz_1, :, :, :]
                aux_sty = aux_outputs[idx][sz_1:sz_2, :, :, :]
                loss_aux_ps += aux_criterion(aux_base, aux_sty)

            # Source Example Losses
            sc_targets = sc_targets + sc_targets
            loss_l_sc, loss_c_sc = criterion_sc(sc_out, sc_targets)
            loss_sc = loss_l_sc + loss_c_sc

            loss_aux_sc = 0.
            # Second half of batch in aux outputs are source items
            sz_1 = sz_2
            sz_2 = sz_1 + sc_images.size(0)
            for idx in range(len(aux_outputs)):
                # l1, l2 = aux_criterion(aux_outputs_ph[idx], aux_outputs_st[idx])
                aux_base = aux_outputs[idx][sz_1:sz_2, :, :, :]
                aux_sty = aux_outputs[idx][sz_2:, :, :, :]
                loss_aux_sc += aux_criterion(aux_base, aux_sty)
            loss_sc += loss_aux_sc

            # Combine losses
            # print('{:.3f} -- {:.3f} -- {:.3f} -- {:.3f}'.format(loss_ps, loss_aux_ps, loss_sc, loss_aux_sc))
            loss = loss_ps + loss_sc + loss_aux_ps + loss_aux_sc
            loss /= 2
            loss.backward()
            optimizer.step()

            if iteration % log_freq == 0:
                writer.add_scalar("Loss_PS", loss_ps.item(), iteration)
                writer.add_scalar("Conf_loss_PS", loss_c_ps.item(), iteration)
                writer.add_scalar("Loc_loss_PS", loss_l_ps.item(), iteration)
                writer.add_scalar("Style_loss", loss_aux_ps, iteration)

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                save_path = os.path.join(output_path, 'weights', 'iteraton-{}.pth'.format(iteration))
                torch.save(model.state_dict(), save_path)

            if iteration >= max_iter - (10*test_freq):
                if iteration == max_iter or iteration % test_freq == 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    accuracy_history.append(eval.evaluate(model, val_dataset,
                                                          os.path.join(output_path, 'iteration-{}'.format(iteration))))
                    model.train()

                    writer.add_scalar("Test_mAP", accuracy_history[-1]['mAP'], iteration)
                    # keep best  model
                    if accuracy_history[-1]['mAP'] > best_map:
                        best_map = accuracy_history[-1]['mAP']
                        best_model = copy.deepcopy(model.state_dict())

                    # summary writer
            else:
                if not iteration % 1000 and iteration > 0:
                    model.eval()
                    eval.FULL_REPORT = False
                    acc = eval.evaluate(model, val_dataset, os.path.join(output_path, 'tempdets'.format(iteration)))
                    writer.add_scalar("Test_mAP", acc['mAP'], iteration)
                    model.train()
                    # summary writer

            iteration += 1
    return model, best_model, best_map, accuracy_history


def boxes_from_dets(detections, thresh=0.25):
    targets = []
    confs = []
    for bidx in range(detections.size(0)):
        pthresh = thresh
        while True:
            bidx_targets = None
            bidx_scores = None
            for cidx in range(1, detections.size(1)):
                    scores = detections[bidx, cidx, :, 0].squeeze()
                    bboxes = detections[bidx, cidx, :, 1:]

                    # scores = scores.view(-1, 1)
                    lookup = scores > pthresh
                    if lookup.sum() > 0:
                        scores = scores[lookup]
                        bboxes = bboxes[lookup, :]
                        lbls = torch.ones(bboxes.size(0), 1) * (cidx - 1)
                        bboxes = torch.cat((bboxes, lbls), 1)
                        if bidx_targets is None:
                            bidx_targets = bboxes
                            bidx_scores = scores
                        else:
                            bidx_targets = torch.cat((bidx_targets, bboxes), 0)
                            bidx_scores = torch.cat((bidx_scores, scores), 0)

            if bidx_targets is None:
                pthresh *= 0.9
            else:
                break
        targets.append(bidx_targets)
        confs.append(bidx_scores)
    return confs, targets


def visualize_boxes(images, style_ims, targets, image_base=None):
    count = 0
    if image_base is None:
        image_base = 'test_imgs'

    for bidx in range(images.size(0)):

        image_out = os.path.join(image_base, 'example_{}.jpg'.format(count))
        count += 1

        if not os.path.exists(os.path.dirname(image_out)):
            os.makedirs(os.path.dirname(image_out))

        bidx_anno = targets[bidx].tolist()
        annos = []
        for anno in bidx_anno:
            lbl = int(anno[-1])
            anno = [i*images.size(-1) for i in anno[:-1]]
            anno.append(lbl)
            annos.append(anno)
        if style_ims is None:
            export_images([images[bidx, :, :, :]], image_out, annos=annos)
        else:
            export_images([images[bidx, :, :, :], style_ims[bidx, :, :, :]], image_out,
                          annos=annos)
    pass
