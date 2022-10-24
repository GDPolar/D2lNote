# %matplotlib inline
import torch
from d2l import torch as d2l

torch.set_printoptions(2)  # 精简输出精度

data = torch.rand(size=(1, 3, 100, 80)) # 构造输⼊数据，高 100，宽 80
sizes=[0.75, 0.5, 0.25]
ratios=[1, 2, 0.5]

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    # data 的后两位，也就是图像的高宽
    in_height, in_width = data.shape[-2:]

    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios) # 3, 3

    # 为了减少计算复杂度，只考虑包含 s1（num_sizes 个）或 r1（num_ratios - 1 个） 的组合
    boxes_per_pixel = (num_sizes + num_ratios - 1) 
    # list 转为 tensor
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为将锚点移动到像素的中心，设置偏移量。
    # 一个像素高为 1 且宽为 1，选择偏移 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长，steps_h = 1 / 100 = 0.01
    steps_w = 1.0 / in_width  # 在x轴上缩放步长，steps_w = 1 / 80 = 0.0125

    # 通过 torch.meshgrid() 函数生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # eg:
    # center_h： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # center_w： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # shift_y tensor([
    #               [0.1250, 0.1250, 0.1250, 0.1250],
    #               [0.3750, 0.3750, 0.3750, 0.3750],
    #               [0.6250, 0.6250, 0.6250, 0.6250],
    #               [0.8750, 0.8750, 0.8750, 0.8750]]) 
    # shift_x tensor([
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750]])

    # 展平
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    # 全部每个像素中心点坐标
    # tensor([0.1250, 0.1250, 0.1250, 0.1250, 0.3750, 0.3750, 0.3750, 0.3750, 0.6250,
    #     0.6250, 0.6250, 0.6250, 0.8750, 0.8750, 0.8750, 0.8750]) 
    # tensor([0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750, 0.1250,
    #     0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750])#

    # 生成 boxes_per_pixel 个高和宽，即 num_sizes + num_ratios - 1
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 除以2来获得半高和半宽(半高半宽和中心坐标相加减后，就得到右下和左上的坐标)
    # 最后整个 anchor_manipulations 的 shape 为 [8000*5,4]
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w) # 561 728
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape # torch.Size([1, 2042040, 4])

# 输出图像中任意一个像素点5个锚框中的1个
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :] # print(0.06,0.07,0.63,0.82)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        #如果参数里面没给obj，就用后面default_values指定的，例如下面的color就是这个意思
        if obj is None:  
            obj = default_values
            
        # 判断obj的类型是不是我们制定的
        # 补充：list是列表，tuple是元组(元组是有序列表，并且一旦初始化就不可以再修改了)
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        #下面的是用来进入text文字描述的
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 前面处理中对锚框的宽高进行了归一化，所以绘制锚框时，要恢复原始的坐标值
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    # boxes1(左上x，左上y，右下x，右下y)
    # eg：boxes1 [1,1,3,3],[0,0,2,4],[1,2,3,4]] 
    #     boxes2[[0,0,3,3],[2,0,5,2]]
    # 计算一个框的面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # 重叠部分左上角坐标（取最大的值）
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # boxes1：tensor([[1, 1, 3, 3],
    #               [0, 0, 2, 4],
    #               [1, 2, 3, 4]])
    # boxes2:tensor([[0, 0, 3, 3],
    #               [2, 0, 5, 2]])
    # 由于维度不同，要用广播机制，真正计算的时候，是下面这样的
    # tensor([[[1, 1],[1, 1]],
    #         [[0, 0],[0, 0]],
    #         [[1, 2],[1, 2]]])
    # tensor([[0, 0],[2, 0]]
    #        [[0, 0],[2, 0]]
    #        [[0, 0],[2, 0]])
    # 此时inter_upperlefts 为：
    # tensor([[[1, 1],
    #         [2, 1]],
    #        [[0, 0],
    #        [2, 0]],
    #       [[1, 2],
    #        [2, 2]]])
    # 重叠部分右下坐标（取最大的值），同上面的过程，就不再赘述了
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # clamp(min=0)用来限制inters最小不能低于0
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 又用了一次广播机制
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，我们标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)