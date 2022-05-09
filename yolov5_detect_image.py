import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import itertools


class Y5Detect:
    def __init__(self, weights):
        """
        :param weights: 'yolov5s.pt'
        """
        self.weights = weights
        self.model_image_size = 640
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.model, self.device = self.load_model(use_cuda=False)

        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(self.model_image_size, s=stride)
        self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def load_model(self, use_cuda=False):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')

        model = attempt_load(self.weights, map_location=device)
        return model, device

    def preprocess_image(self, image_rgb):
        # Padded resize
        img = letterbox(image_rgb.copy(), new_shape=self.image_size)[0]

        # Convert
        img = img.transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, image_rgb):
        image_rgb_shape = image_rgb.shape
        img = self.preprocess_image(image_rgb)
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred,
                                   self.conf_threshold,
                                   self.iou_threshold,)
        bboxes = []
        labels = []
        scores = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_rgb_shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    with torch.no_grad():
                        x1 = xyxy[0].cpu().data.numpy()
                        y1 = xyxy[1].cpu().data.numpy()
                        x2 = xyxy[2].cpu().data.numpy()
                        y2 = xyxy[3].cpu().data.numpy()
                        #                        print('[INFO] bbox: ', x1, y1, x2, y2)
                        bboxes.append(list(map(int, [x1, y1, x2, y2])))
                        label = self.class_names[int(cls)]
                        #                        print('[INFO] label: ', label)
                        labels.append(label)
                        score = conf.cpu().data.numpy()
                        #                        print('[INFO] score: ', score)
                        scores.append(float(score))
        return bboxes, labels, scores


def draw_boxes_detection(image, boxes, scores=None, labels=None, class_names=None, line_thickness=2, font_scale=1.0,
               font_thickness=2):
    num_classes = len(class_names)
    # colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    if scores is not None and labels is not None:
        for b, l, s in zip(boxes, labels, scores):
            if class_names is None:
                class_name = 'person'
                class_id = 0
            elif l not in class_names:
                class_id = int(l)
                class_name = class_names[class_id]
            else:
                class_name = l
                class_id = class_names.index(l)

            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            # color = colors[class_id]
            color = (255, 0, 0)
            label = '-'.join([class_name, score])

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, line_thickness)
            #cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (255, 255, 255), -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                        font_thickness)
    elif labels is not None:
        for b, l in zip(boxes, labels):
            xmin, ymin, xmax, ymax = list(map(int, b))
            idx = class_names.index(l)
            color = colors[idx]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    elif scores is not None:
        idx = 0
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = list(map(int, b))
            score = '{:.4f}'.format(s)
            color = colors[idx]
            label = '-'.join([score])
            idx += 1

            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            # cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        color = (0, 255, 0)
        for b in boxes:
            xmin, ymin, xmax, ymax = list(map(int, b))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
    return image


def draw_bounding_box_ground_truth(image, path_label, list_name_label, line_thickness=2,
                                   font_scale=1.0, font_thickness=2):
    true_box = []
    true_labels = []
    h, w, _ = image.shape
    with open(path_label, mode='r') as file_yolo:
        list_labels = file_yolo.readlines()

    for label in list_labels:
        label = label.strip()
        label = label.split(' ')
        bbox = list(map(float, label))
        xmax, xmin = int((bbox[1] + bbox[3]/2) * w), int((bbox[1] - bbox[3]/2)*w)
        ymax, ymin = int((bbox[2] + bbox[4]/2) * h), int((bbox[2] - bbox[4] / 2) * h)
        ret, baseline = cv2.getTextSize(list_name_label[int(bbox[0])],
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
        #cv2.rectangle(image, (xmin, ymin - ret[1] - baseline), (xmin + ret[0], ymin), (255, 255, 255), -1)
        cv2.putText(image, list_name_label[int(bbox[0])] , (xmin, ymin - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    font_thickness)
        true_box.append([xmin, ymin, xmax, ymax])
        true_labels.append(list_name_label[int(bbox[0])])
    return image, true_box, true_labels


def cal_iou(true_box, pre_box):
    """
    function: calculator IOU value
    :param true_box: list bbox true
    :param pre_box: list bbox predict
    :return: IOU value is in the range 0-1
    """
    xmin_true_box, ymin_true_box, xmax_true_box, ymax_true_box = true_box
    xmin_pre_box, ymin_pre_box, xmax_pre_box, ymax_pre_box = pre_box
    if xmax_pre_box < xmin_true_box or ymax_pre_box < ymin_true_box:
        return 0.0
    if xmin_pre_box > xmax_true_box or ymin_pre_box > ymax_true_box:
        return 0.0
    True_box_area = (xmax_true_box - xmin_true_box + 1)*(ymax_true_box - ymin_true_box + 1)
    Pre_box_area = (xmax_pre_box - xmin_pre_box + 1)*(ymax_pre_box - ymin_pre_box + 1)
    xmin_inter = np.max([xmin_pre_box, xmin_true_box])
    ymin_inter = np.max([ymin_pre_box, ymin_true_box])
    xmax_inter = np.min([xmax_pre_box, xmax_true_box])
    ymax_inter = np.min([ymax_pre_box, ymax_true_box])
    Intersection_area = (xmax_inter - xmin_inter + 1)*(ymax_inter - ymin_inter + 1)
    Union_area = (True_box_area + Pre_box_area - Intersection_area)
    return Intersection_area / Union_area


def check_dectect_and_save(true_bbox, pre_bbox, true_labels, pre_labels, path_save):
    """
    function: check and save predict result
    :param true_bbox: list bbox true
    :param pre_bbox: list bbox predict
    :param true_labels: list name label true
    :param pre_labels: list name label predict
    :return:
    """
    len_truebox = len(true_bbox)
    len_prebbox = len(pre_bbox)
    list_tick_truebox = []
    list_tick_prebox = []

    counter = 0

    #recognize labels classification
    for i_truebox in range(len_truebox):
        #check = 0
        for i_prebox in range(len_prebbox):
            iou = cal_iou(true_bbox[i_truebox], pre_bbox[i_prebox])
            if iou > 0.45:
                list_tick_truebox.append(i_truebox)
                list_tick_prebox.append(i_prebox)
                #if check == 1:
                #    continue
                counter += 1
                list_confusion_matrix_gt.append(name_label.index(true_labels[i_truebox]))
                list_confusion_matrix_pre.append(name_label.index(pre_labels[i_prebox]))
                continue
                if not os.path.exists(path_save + 'Nhận diện được'):
                    os.mkdir(path_save + 'Nhận diện được')
                path = path_save + 'Nhận diện được/' + true_labels[i_truebox]
                path1 = path + '/' + pre_labels[i_prebox]
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(path1):
                    os.mkdir(path1)
                cv2.imwrite(path1 + '/' + name_image + '_' + str(counter) + '.jpg', arr_image)
                #check = 1
    """
    # missing identification labels classification
    counter = 0

    for i_truebox in range(len_truebox):
        if i_truebox not in list_tick_truebox:
            list_confusion_matrix_gt.append(name_label.index(true_labels[i_truebox]))
            list_confusion_matrix_pre.append(len(name_label))
            if not os.path.exists(path_save + 'Nhận diện thiếu'):
                os.mkdir(path_save + 'Nhận diện thiếu')
            path = path_save + 'Nhận diện thiếu/' + true_labels[i_truebox]
            if not os.path.exists(path):
                os.mkdir(path)
            counter += 1
            path_save_1 = path + '/' + name_image+ '_' + str(counter) + '.jpg'
            cv2.imwrite(path_save_1, arr_image)

    #classification of redundant identity labels
    counter = 0
    if len(list_tick_prebox) != len_prebbox:
        for i_prebox in range(len_prebbox):
            if i_prebox not in list_tick_prebox:
                list_confusion_matrix_gt.append(len(name_label))
                list_confusion_matrix_pre.append(name_label.index(pre_labels[i_prebox]))
                if not os.path.exists(path_save + 'Nhận diện dư'):
                    os.mkdir(path_save + 'Nhận diện dư')
                path = path_save + 'Nhận diện dư/' + pre_labels[i_prebox]
                if not os.path.exists(path):
                    os.mkdir(path)
                counter += 1
                path_save_2 = path + '/' + name_image + '_' + str(counter) + '.jpg'
                cv2.imwrite(path_save_2, arr_image)
    """


def plot_confusion_matrix(cm, classes, path_save,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    sum_c = cm.sum(axis=0, keepdims=True)
    cm = cm.astype('float')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if sum_c[0][j] == 0:
                cm[i][j] = cm[i][j]
            else:
                cm[i][j] = 100 * cm[i][j] / sum_c[0][j]
                for a in range(10, 1, -1):
                    cm[i][j] = round(cm[i][j], a)

    #cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    classes1 = np.copy(classes)
    #classes1[-1] = 'Background FP'
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #aspect='auto'
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes1, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(path_save + 'confusionMatrix-percent.png', bbox_inches="tight")


def my_confusion_matrix(y_true, y_pred, name_labels):
    N = len(name_labels)
    cm = np.zeros((N, N), dtype='uint32')
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1
    cm = cm.T
    return cm


def draw_plot_confusion_matrix(y_true, y_pred, name_labels, path_save):

   # name_labels.append('Background FN')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cnf_matrix = my_confusion_matrix(y_true, y_pred, name_labels)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=name_labels,
                          path_save=path_save, title='Confusion matrix - Percent')


global name_label, list_confusion_matrix_gt, list_confusion_matrix_pre, name_image, arr_image

if __name__ == '__main__':

    path_data = "/home/duyngu/Downloads/dataset_2/dataset_1/val/"
    path_save = '/home/duyngu/Desktop/result/'
    y5_model = Y5Detect(weights='/home/duyngu/Downloads/model_5/content/mydrive/MyDrive/train/exp/weights/best.pt')
    name_label = y5_model.class_names
    list_confusion_matrix_gt = []
    list_confusion_matrix_pre = []
    list_label = glob(path_data + "*.txt")
    list_image_test = glob(path_data + "*.jpg")
    list_label = sorted(list_label)
    list_image_test = sorted(list_image_test)
    for i in range(len(list_image_test)):
        name_image = list_image_test[i].split('/')[-1].split('.')[0]
        image_bgr = cv2.imread(list_image_test[i])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bboxes, labels, scores = y5_model.predict(image_rgb)
        if bboxes is not None:
            image_bgr, true_bbox, true_labels = draw_bounding_box_ground_truth(image_bgr,
                                                                               list_label[i],
                                                                               y5_model.class_names)

            draw_boxes_detection(image_bgr, bboxes, scores=scores,
                                 labels=labels, class_names=y5_model.class_names)
            arr_image = np.copy(image_bgr)
            check_dectect_and_save(true_bbox=true_bbox, pre_bbox=bboxes, true_labels=true_labels,
                                   pre_labels=labels, path_save=path_save)
        print('Saved' + str(i + 1))
    draw_plot_confusion_matrix(y_true=list_confusion_matrix_gt,
                               y_pred=list_confusion_matrix_pre,
                               name_labels=name_label, path_save=path_save)
