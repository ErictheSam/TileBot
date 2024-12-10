import sys
import os
sys.path.append(os.path.dirname(__file__) +'/../')
import cv2
import torch
import json
import tqdm
import argparse
import numpy as np
from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.models.build_model import build_model
from mlsd_pytorch.data.utils import deccode_lines
from mlsd_pytorch.metric import msTPFP, AP
import mlsd_pytorch.utils_new as utils_new
#from albumentations import Normalize
current_dir =os.path.abspath(os.path.dirname(__file__))
print(current_dir)
device = 3


def decode_lines(displacement, scores, indices, w=280, score_thresh = 0.1, len_thresh=2):
    # print(tpMap.shape)
    # displacement = tpMap[:, 1:5, :, :]
    valid_inx = torch.where(scores > score_thresh)
    scores = scores[valid_inx]
    indices = indices[valid_inx]

    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    center_ptss = torch.cat((xx, yy),dim=-1)
    start_point = center_ptss + displacement[0, :2, yy, xx].reshape(2, -1).permute(1,0)
    end_point = center_ptss + displacement[0, 2:, yy, xx].reshape(2, -1).permute(1,0)

    lines = torch.cat((start_point, end_point), dim=-1)

    all_lens = (end_point - start_point) ** 2
    all_lens = all_lens.sum(dim=-1)
    all_lens = torch.sqrt(all_lens)

    valid_inx = torch.where(all_lens > len_thresh)

    center_ptss = center_ptss[valid_inx]
    lines = lines[valid_inx]
    scores = scores[valid_inx]

    return lines

def get_args():
    args = argparse.ArgumentParser()
    print(current_dir)
    args.add_argument("--config", type=str,default = current_dir  + '/configs/enet.yaml')
    args.add_argument("--model_path", type=str,
                      default= current_dir +"/workdir/models/general_origin/best.pth")
    args.add_argument("--gt_json", type=str,
                      default= current_dir +"/data/wireframe_raw/valid.json")
    args.add_argument("--img_dir", type=str,
                      default= current_dir + "/data/wireframe_raw/images_lzj/")
    args.add_argument("--sap_thresh", type=float, help="sAP thresh", default=20.0)
    args.add_argument("--top_k", type=float, help="top k lines", default= 200)
    args.add_argument("--min_len", type=float, help="min len of line", default=20)
    args.add_argument("--score_thresh", type=float, help="line score thresh", default=0.02)
    args.add_argument("--input_size", type=int, help="image input size", default=280)
    args.add_argument("--video_capture", type=str, default=current_dir +"/output2_new.avi")
    return args.parse_args()

#test_aug = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
func = torch.nn.Sigmoid()


def infer_cap(cap_dir, out_dir, model, input_size=280, score_thresh=0.03, min_len=25, topk=300):
    cap = cv2.VideoCapture(cap_dir)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter(out_dir,fourcc, 15,(280,280))
    with torch.no_grad():
        outj = 0
        while cap.isOpened():
            ret, image = cap.read()
            
            if(ret == False):
                break
            print(image.shape)
            h = int(image.shape[0]/2)
            w = int(image.shape[1]/2)
            stride = 160#int(input_size/2)
            image = image[h-stride:h+stride,w-stride:w+stride]
            image = cv2.resize(image,(input_size, input_size))
            image = cv2.convertScaleAbs(image, alpha=5.0)
            clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(14,14))
            # image = cv2.resize(image,(320,320))
            # image = cv2.medianBlur(image,5)
            planes = cv2.split(image)
            gm = []
            for i in range(3):
                gm.append(clahe.apply(planes[i]))
            image = cv2.merge(gm)
            img = image.copy()
            image = image / 255.0
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().cuda(3)
            # print(image.shape)
            batch_outputs = model(image)
            # print(img_re)
            pre_img = batch_outputs[:,0,:,:]
            #pre_img = img_re.squeeze(0)
            img_re = pre_img.permute(1, 2, 0).cpu().numpy()
            img_re = img_re * 255
            img_re = cv2.cvtColor(img_re.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            tp_mask = batch_outputs[:, 8:, :, :]
            #print(tp_mask[:,0,:,:], pre_img)
            center_map = tp_mask[:,0,:,:]
            center_map = torch.sigmoid(center_map)
            center_map *= pre_img
            tp_mask[:,0,:,:] = center_map
            center_map = center_map.permute(1, 2, 0).cpu().numpy()
            l = np.max(center_map)
            center_map = (center_map / l) * 255.0
            center_map = cv2.cvtColor(center_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # print(center_map)
            # print(center_map)
            #_, center_map = cv2.threshold(center_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
            center_ptss, pred_lines, scores = deccode_lines(tp_mask, score_thresh, min_len, topk, 5)
            # print(center_ptss.shape)
            pred_lines = pred_lines.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            pred_lines_list = []
            scores_list = []
            for line, score in zip(pred_lines, scores):
                x0, y0, x1, y1 = line
                if((x1 - x0) ** 2 +(y1 - y0) ** 2) > 900:
                    pred_lines_list.append([x0, y0, x1, y1])
                    scores_list.append(score)
            for l in pred_lines_list:
                cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,200,200), 5,16)
                cv2.line(center_map, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,200,200), 5,16)
            
            print(img.shape)
            #out.write(img)
            outj += 1
            print(outj)
            cv2.imwrite(current_dir+'/pics_new/'+str(outj)+".png", img)
            cv2.imwrite(current_dir+'/pics_new/'+str(outj)+"center.png", center_map)
    #out.release()
    cap.release()

def infer_one(img_fn, model, input_size=280, score_thresh=0.02, min_len=5, topk=300):
    img = cv2.imread(img_fn)
    # img = cv2.resize(img, (input_size, input_size))
    # # img_re = np.vstack((img_re,img_re,img_re))
    # _, img_re = cv2.threshold(img_re, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
    # img_re = cv2.cvtColor(img_re,cv2.COLOR_GRAY2BGR)
    
    clahe= cv2.createCLAHE(clipLimit=1.0, tileGridSize=(14,14))
    h, w, _ = img.shape 
    img = img[h//2-140:h//2+140,w//2-140:w//2+140]
    h, w, _ = img.shape 
    image = img / 255.0
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = cv2.bilateralFilter(image,5,150,150)
    # planes = cv2.split(image)
    # gm = []
    # for i in range(3):
    #     k = planes[i]
    #     # k = cv2.normalize(k, None, 0, 255, cv2.NORM_MINMAX)
    #     gm.append(clahe.apply(k))
    # image = cv2.merge(gm)
    # image = cv2.medianBlur(image,5)
    # img = image.copy()
    # image = image / 255.0
    # 
    #img = (img / 127.5) - 1.0
    # img = test_aug(image=img)['image']
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().cuda(device)

    with torch.no_grad():
        batch_outputs, scores, indices = model(image)
        # print(img_re)
        pre_img = batch_outputs[:,0,:,:]
        #pre_img = img_re.squeeze(0)
        img_re = pre_img.permute(1, 2, 0).cpu().numpy()
        img_re = img_re * 255
        # img_re = img_re.astype(np.uint8)
        # print(img_re)
        img_re = cv2.normalize(img_re, None, 0, 255, cv2.NORM_MINMAX,  dtype=cv2.CV_8UC1)
        # _, img_re = cv2.threshold(img_re, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        center_map = torch.sigmoid(batch_outputs[:,1,:,:]).permute(1, 2, 0).cpu().numpy()
        l = np.max(center_map)
        center_map = (center_map/l) * 255
        # center_map = cv2.normalize(center_map, None, 0, 255, cv2.NORM_MINMAX)
        # img_re = cv2.cvtColor(img_re, cv2.COLOR_GRAY2BGR)
    tp_mask = batch_outputs[:, 2:, :, :]
    tp_mask = tp_mask.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    pred_linelist, _ = utils_new.pred_lines(img_re, tp_mask[0], scores, indices, score_thresh, min_len)
    #pred_lines = pred_lines.detach().cpu().numpy()
    #scores = scores.detach().cpu().numpy()

    pred_lines_list = []
    scores_list = []
    for line in pred_linelist:
        print(line)
        x0, y0, x1, y1 = line[0], line[1], line[2], line[3]

        x0 = w * x0 / input_size
        x1 = w * x1 / input_size

        y0 = h * y0 / input_size
        y1 = h * y1 / input_size

        pred_lines_list.append([x0, y0, x1, y1])
        #scores_list.append(score)
    for l in pred_lines_list:
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,0,255), 3, 16)
        # cv2.line(center_map, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0,200,200), 1,16)
    print("write")
    cv2.imwrite(current_dir+'/written_lzj_ori/'+os.path.basename(img_fn), img)
    cv2.imwrite(current_dir+'/written_lzj_seg_ori/'+os.path.basename(img_fn), img_re)
    cv2.imwrite(current_dir+'/written_lzj_center_ori/'+os.path.basename(img_fn), center_map)
    return {
        'full_fn': img_fn,
        'filename': os.path.basename(img_fn),
        'width': w,
        'height': h,
        'lines': pred_lines_list
        #'scores': scores_list
    }


def calculate_sAP(gt_infos, pred_infos, sap_thresh):
    assert len(gt_infos) == len(pred_infos)

    tp_list, fp_list, scores_list = [], [], []
    n_gt = 0

    for gt, pred in zip(gt_infos, pred_infos):
        assert gt['filename'] == pred['filename']
        h, w = gt['height'], gt['width']
        pred_lines = np.array(pred['lines'], np.float32)
        pred_scores = np.array(pred['scores'], np.float32)

        gt_lines = np.array(gt['lines'], np.float32)
        scale = np.array([128.0/ w, 128.0/h, 128.0/ w, 128.0/h], np.float32)
        pred_lines_128 = pred_lines * scale
        gt_lines_128 = gt_lines * scale

        tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)

        n_gt += gt_lines_128.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(pred_scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx]) / n_gt
    fp = np.cumsum(fp_list[idx]) / n_gt
    rcs = tp
    pcs = tp / np.maximum(tp + fp, 1e-9)
    sAP = AP(tp, fp) * 100

    return  sAP


def main(args):
    cfg = get_cfg_defaults()
    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    print('using config: ', args.config.strip())
    cfg.merge_from_file(args.config)
    model_device = torch.device("cuda:"+str(device))
    model = build_model(cfg, ).eval()
    model.load_state_dict(torch.load(args.model_path, map_location=model_device), strict=True)
    # model.enet_backbone.load_state_dict(torch.load(cfg.train.enet_load_from, map_location=model_device),strict=False)
    model = model.to(model_device)
    label_file = args.gt_json
    img_dir = args.img_dir
    contens = json.load(open(label_file, 'r'))

    gt_infos = []
    pred_infos = []
    
    # infer_cap(args.video_capture, "out.avi", model, args.input_size, args.score_thresh, args.min_len, args.top_k)
    # for c in tqdm.tqdm(contens):
    #     gt_infos.append(c)
    #     fn = c['filename'][:-4] + '.jpg'
    #     full_fn = img_dir + fn
    #     pred_infos.append(infer_one(full_fn, model,
    #                                 args.input_size,
    #                                 args.score_thresh,
    #                                 args.min_len, args.top_k ))

    img_list = os.listdir(img_dir)
    for img in img_list:
        full_fn = img_dir + img
        pred_infos.append(infer_one(full_fn, model,
                                    args.input_size,
                                    args.score_thresh,
                                    args.min_len, args.top_k ))

    print(pred_infos)
    # # ap = calculate_sAP(gt_infos, pred_infos, args.sap_thresh)
    
    # new_one = '/home/syb/unet/Unet-pytorch/generalized_mask'
    # for h in os.listdir(new_one):
    #     infer_one(new_one + '/' + h, model, args.input_size,
    #                                 args.score_thresh,
    #                                 args.min_len, args.top_k  )


if __name__ == '__main__':
    main(get_args())
