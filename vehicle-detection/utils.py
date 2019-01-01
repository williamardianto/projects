import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import logging
import struct # get_image_size
import imghdr # get_image_size
import cv2

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0)
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0)
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0)
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea
# check nearby box
def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = int(len(anchors)/num_anchors)
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
    # grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(-1)
    # grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(-1)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
    # anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    # anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')
    return all_boxes
#
# def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
#     import cv2
#     colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
#     def get_color(c, x, max_val):
#         ratio = float(x)/max_val * 5
#         i = int(math.floor(ratio))
#         j = int(math.ceil(ratio))
#         ratio = ratio - i
#         r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
#         return int(r*255)
#
#     width = img.shape[1]
#     height = img.shape[0]
#     for i in range(len(boxes)):
#         box = boxes[i]
#         x1 = int(round((box[0] - box[2]/2.0) * width))
#         y1 = int(round((box[1] - box[3]/2.0) * height))
#         x2 = int(round((box[0] + box[2]/2.0) * width))
#         y2 = int(round((box[1] + box[3]/2.0) * height))
#
#         xc = int(box[0] * width)
#         yc = int(box[1] * height)
#
#         if color:
#             rgb = color
#         else:
#             rgb = (255, 0, 0)
#         if len(box) >= 7 and class_names:
#             cls_conf = box[5]
#             cls_id = box[6]
#             print('%s: %f' % (class_names[cls_id], cls_conf))
#             classes = len(class_names)
#             offset = cls_id * 123457 % classes
#             red   = get_color(2, offset, classes)
#             green = get_color(1, offset, classes)
#             blue  = get_color(0, offset, classes)
#             if color is None:
#                 rgb = (red, green, blue)
#             img = cv2.putText(img, class_names[cls_id], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
#         img = cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 2)
#         # img = cv2.circle(img,(xc,yc), 2, rgb, -1)
#     if savename:
#         print("save plot results to %s" % savename)
#         cv2.imwrite(savename, img)
#     # return img

def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)

        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5, 5) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=0):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray: # cv2 image
        img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    #for j in range(100):
    #    sys.stdout.write('%f ' % (output.storage()[j]))
    #print('')
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)[0]
    #for j in range(len(boxes)):
    #    print(boxes[j])
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets
      
def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count



def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24: 
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2 
                ftype = 0 
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2 
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height


def box_convert(bboxes, width, height):
    bboxes_center=[]
    for i in range(len(bboxes)):
        box = bboxes[i]
        x = (box[0] - box[2]/2.0) * width
        y = (box[1] - box[3]/2.0) * height
        w = ((box[0] + box[2]/2.0) * width) - x
        h = ((box[1] + box[3]/2.0) * height) - y
        xc = box[0] * width
        yc = box[1] * height
        bboxes_center.append({'object': ((round(x),round(y),round(w),round(h)),(round(xc),round(yc))), 'bboxes': box})
    return bboxes_center

# def logging(message):
#     print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

###########################################

def distance(x, y, type='euclidian', x_weight=1.0, y_weight=1.0):
    if type == 'euclidian':
        return math.sqrt(float((x[0] - y[0])**2) / x_weight + float((x[1] - y[1])**2) / y_weight)

class VehicleCounter():
    '''
        Counting vehicles that entered in exit zone.

        Purpose of this class based on detected object and local cache create
        objects pathes and count that entered in exit zone defined by exit masks.

        exit_masks - list of the exit masks.
        path_size - max number of points in a path.
        max_dst - max distance between two points.
    '''

    def __init__(self, exit_masks=[], class_names=[], path_size=10, max_dst=30, x_weight=1.0, y_weight=1.0, save=True):
        super(VehicleCounter, self).__init__()

        self.exit_masks = exit_masks

        self.vehicle_count = 0
        self.vehicle_speed = 0
        self.event = 0
        self.class_idx_2_name = {i: n for i, n in enumerate(class_names)}
        self.vehicle_class = {name: 0 for name in class_names}


        self.path_size = path_size
        self.pathes = []
        self.max_dst = max_dst
        self.x_weight = x_weight
        self.y_weight = y_weight
        self.save = save
        self.tracking_points = []
        self.log = logging.getLogger(self.__class__.__name__)

    def check_exit(self, point):
        for exit_mask in self.exit_masks:
            try:
                if exit_mask[point[1]][point[0]] == 255:
                    return True
            except:
                return True
        return False

    def __call__(self, context):
        objects = context['objects']
        context['exit_masks'] = self.exit_masks
        context['pathes'] = self.pathes
        context['vehicle_count'] = self.vehicle_count
        context['vehicle_speed'] = self.vehicle_speed
        context['vehicle_class'] = self.vehicle_class


        if not objects:
            return context

        # points = np.array(objects)[:, 0:2]
        # points = points.tolist()

        # add new points if pathes is empty
        if not self.pathes:
            # for match in points:
            #     self.pathes.append({'path': [match], 'event': 0})
            for match in objects:
                self.pathes.append({'path': [{'object': match['object'], 'bboxes': match['bboxes']}], 'event': 0})

        else:
            # link new points with old pathes based on minimum distance between
            # points
            new_pathes = []

            for path in self.pathes:
                _min = 999999
                _match = None
                for p in objects:
                    if len(path['path']) == 1:
                        # distance from last point to current
                        d = distance(p['object'][1], path['path'][-1]['object'][1])
                    else:
                        # based on 2 prev points predict next point and calculate
                        # distance from predicted next point to current
                        xn = 2 * path['path'][-1]['object'][1][0] - path['path'][-2]['object'][1][0]
                        yn = 2 * path['path'][-1]['object'][1][1] - path['path'][-2]['object'][1][1]
                        d = distance(
                            p['object'][1], (xn, yn),
                            x_weight=self.x_weight,
                            y_weight=self.y_weight
                        )

                    if d < _min:
                        _min = d
                        _match = p

                if _match and _min <= self.max_dst:
                    objects.remove(_match)
                    path['path'].append(_match)
                    new_pathes.append(path)

                # do not drop path if current frame has no matches
                if _match is None:
                    new_pathes.append(path)

            self.pathes = new_pathes

            # add new pathes
            if len(objects):
                print(len(objects))
                for p in objects:
                    # do not add points that already should be counted
                    if self.check_exit(p['object'][1]):
                        continue
                    self.pathes.append({'path': [p], 'event': 0})

        # save only last N points in path
        for i, _ in enumerate(self.pathes):
            # select the last 10 points
            self.pathes[i]['path'] = self.pathes[i]['path'][self.path_size * -1:]

        # count vehicles and drop counted pathes:
        new_pathes = []
        for i, path in enumerate(self.pathes):
            d = path['path'][-2:]

            #check event
            if(len(d) >= 2 and self.path_size <= len(path['path'])):
                x_list = [i['object'][1][0] for i in path['path']]
                y_list = [i['object'][1][1] for i in path['path']]

                xy_list = list(zip(x_list, y_list))
                points_dist = []
                for i in range(len(xy_list)-1):
                    points_dist.append(distance(xy_list[i], xy_list[i+1]))

                mean_dist = np.mean(points_dist)
                pred_speed = int((mean_dist * 6) + 5)
                if pred_speed <= 10:
                    path['event'] = 1
                else:
                    path['event'] = 0

            if (
                # need at list two points to count
                len(d) >= 2 and
                # prev point not in exit zone
                not self.check_exit(d[0]['object'][1]) and
                # current point in exit zone
                self.check_exit(d[1]['object'][1]) and
                # path len is bigger then min
                self.path_size <= len(path['path'])
            ):
                self.vehicle_count += 1
                path_classes = [o['bboxes'][6] for o in path['path']]
                class_in_path = max(path_classes, key=path_classes.count)
                self.vehicle_class[self.class_idx_2_name[class_in_path]] += 1

                x_list = [i['object'][1][0] for i in path['path']]
                y_list = [i['object'][1][1] for i in path['path']]

                xy_list = list(zip(x_list, y_list))
                points_dist = []
                for i in range(len(xy_list)-1):
                    points_dist.append(distance(xy_list[i], xy_list[i+1]))

                mean_dist = np.mean(points_dist)
                self.vehicle_speed = int((mean_dist * 6) + 5)

                # if self.save:
                #     xy_list = [i[0][:2] for i in path]
                #     self.tracking_points.append([item for sublist in xy_list for item in sublist])
                #     if self.vehicle_count == 600:
                #         with open('tracking_points.csv', 'w', newline='') as csvfile:
                #             writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                #             writer.writerows(self.tracking_points)
            else:
                # prevent linking with path that already in exit zone
                add = True
                for p in path['path']:
                    if self.check_exit(p['object'][1]):
                        add = False
                        break
                if add:
                    new_pathes.append(path)

        self.pathes = new_pathes

        context['pathes'] = self.pathes
        context['objects'] = objects
        context['vehicle_count'] = self.vehicle_count
        context['vehicle_speed'] = self.vehicle_speed
        context['vehicle_class'] = self.vehicle_class

        self.log.debug('#VEHICLES FOUND: %s' % self.vehicle_count)

        return context


DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)
CAR_COLOURS = [(0, 0, 255)]
EXIT_COLOR = (66, 183, 42)

class Visualizer():
    def __init__(self, save_image=False, image_dir='images'):
        super(Visualizer, self).__init__()

        self.save_image = save_image
        self.image_dir = image_dir
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.video_writer = cv2.VideoWriter('/home/william/project/car_detection2/car_detection_output.mp4', self.fourcc, 20.0, (1280, 720))

    def check_exit(self, point, exit_masks=[]):
        for exit_mask in exit_masks:
            if exit_mask[point[1]][point[0]] == 255:
                return True
        return False

    def draw_pathes(self, img, pathes):
        if not img.any():
            return

        for i, path in enumerate(pathes):
            path = path['path']
            # path = np.array(path)[:, 1].tolist()
            path = [o['object'][1] for o in path]
            for point in path:
                cv2.circle(img, point, 2, CAR_COLOURS[0], -1)
                cv2.polylines(img, [np.int32(path)], False, CAR_COLOURS[0], 1)

        return img

    def plot_boxes_cv2(self, img, pathes, exit_masks=[], class_names=None, color=None):

        color = None
        colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(pathes)):
            box = pathes[i]['path'][-1]['bboxes']
            event = pathes[i]['event']
            x1 = int(round((box[0] - box[2] / 2.0) * width))
            y1 = int(round((box[1] - box[3] / 2.0) * height))
            x2 = int(round((box[0] + box[2] / 2.0) * width))
            y2 = int(round((box[1] + box[3] / 2.0) * height))

            xc = int(box[0] * width)
            yc = int(box[1] * height)

            if self.check_exit((xc,yc), exit_masks):
                continue

            if color:
                rgb = color
            else:
                rgb = (0, 255, 0)
                red = (0, 0, 255)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                # red = get_color(2, offset, classes)
                # green = get_color(1, offset, classes)
                # blue = get_color(0, offset, classes)
                # if color is None:
                #     rgb = (red, green, blue)
            if event:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), red, 2)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, red, 2)
            else:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
            # img = cv2.circle(img,(xc,yc), 2, rgb, -1)

        return img

    def draw_boxes(self, img, pathes, exit_masks=[]):
        for (i, match) in enumerate(pathes):

            contour, centroid = match[-1][:2]
            if self.check_exit(centroid, exit_masks):
                continue

            x, y, w, h = contour

            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1),
                          BOUNDING_BOX_COLOUR, 1)
            cv2.circle(img, centroid, 2, CENTROID_COLOUR, -1)

        return img

    def draw_ui(self, img, vehicle_count, vehicle_class, vehicle_speed, event, exit_masks=[]):

        # this just add green mask with opacity to the image
        for exit_mask in exit_masks:
            _img = np.zeros(img.shape, img.dtype)
            _img[:, :] = EXIT_COLOR
            mask = cv2.bitwise_and(_img, _img, mask=exit_mask)
            cv2.addWeighted(mask, 1, img, 1, 0, img)

        # drawing top block with counts
        cv2.rectangle(img, (0, 0), (img.shape[1], 90), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, ("Vehicles passed: {total} ".format(total=vehicle_count)), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, ('Speed: {speed}'.format(speed=vehicle_speed)), (300,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, ('Event: {evt}'.format(evt=event)), (450,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        text_posistion = [30,180,330,480]
        for i, c in enumerate(vehicle_class):
            cv2.putText(img, (c+": {total} ".format(total=vehicle_class[c])), (text_posistion[i], 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


        return img

    def __call__(self, context):
        frame = context['frame'].copy()
        # frame_number = context['frame_number']
        pathes = context['pathes']
        exit_masks = context['exit_masks']
        vehicle_count = context['vehicle_count']
        vehicle_speed = context['vehicle_speed']
        vehicle_class = context['vehicle_class']
        class_names = context['class_names']

        event = 1 if sum([e['event'] for e in pathes]) > 0 else 0
        frame = self.draw_ui(frame, vehicle_count, vehicle_class, vehicle_speed, event, exit_masks)
        frame = self.draw_pathes(frame, pathes)
        frame = self.plot_boxes_cv2(frame, pathes, exit_masks, class_names)
        # frame = self.draw_boxes(frame, pathes, exit_masks)

        self.video_writer.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # utils.save_frame(frame, self.image_dir +
        #                  "/processed_%04d.png" % frame_number)

        return context