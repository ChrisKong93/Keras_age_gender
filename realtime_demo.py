"""
Face detection
"""
import os
from time import time

import cv2
import keras.backend.tensorflow_backend as KTF
import numpy as np
from scipy import misc
import tensorflow as tf

import RTSCapture
import align.detect_face
from wide_resnet import WideResNet

os.environ['CUDA_VISIBLE_DEVICES']='-1'
rate = 0.2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = rate  # 指定分配30%空间
sess = tf.Session(config=config)  # 设置session
KTF.set_session(sess)
"""
Singleton class for face recongnition task
"""
WRN_WEIGHTS_PATH = "./pretrained_models/weights.18-4.06.hdf5"
face_size = 64
model = WideResNet(face_size, depth=16, k=8)()
model.load_weights(WRN_WEIGHTS_PATH)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=rate, allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# with tf.Session(config=config) as sess:
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)


def load_and_align_data(img, image_size, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_size = np.asarray(img.shape)[0:2]

    # bounding_boxes shape:(1,5)  type:np.ndarray
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果未发现目标 直接返回
    if len(bounding_boxes) < 1:
        return 0, 0, 0

    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    # det = np.squeeze(bounding_boxes[:,0:4])
    det = bounding_boxes

    det_temp = det

    det[:, 0] = np.maximum(det[:, 0], 0)
    det[:, 1] = np.maximum(det[:, 1], 0)
    det[:, 2] = np.minimum(det[:, 2], img_size[1] - 1)
    det[:, 3] = np.minimum(det[:, 3], img_size[0] - 1)

    det_temp[:, 0] = np.maximum(det_temp[:, 0] - margin / 2, 0)
    det_temp[:, 1] = np.maximum(det_temp[:, 1] - margin / 2, 0)
    det_temp[:, 2] = np.minimum(det_temp[:, 2] + margin / 2, img_size[1] - 1)
    det_temp[:, 3] = np.minimum(det_temp[:, 3] + margin / 2, img_size[0] - 1)
    det_temp = det_temp.astype(int)
    det = det.astype(int)
    crop = []
    for i in range(len(bounding_boxes)):
        w = abs(det[i, 0] - det[i, 2])
        h = abs(det[i, 1] - det[i, 3])
        if w > h:
            D = abs(w - h)
            newx1 = det[i, 0]
            newx2 = det[i, 2]
            newy1 = int(det[i, 1] - D / 2)
            newy2 = int(det[i, 3] + D / 2)
            if newy1 < 0:
                newy1 = 0
            if newy2 >= img.shape[0]:
                newy2 = img.shape[0] - 1
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        else:
            D = abs(w - h)
            newx1 = int(det[i, 0] - D / 2)
            newx2 = int(det[i, 2] + D / 2)
            newy1 = det[i, 1]
            newy2 = det[i, 3]
            if newx1 < 0:
                newx1 = 0
            if newx2 >= img.shape[1]:
                newx2 = img.shape[1] - 1
                # img.shape[0]：图像的垂直尺寸（高度）
                # img.shape[1]：图像的水平尺寸（宽度）
                # img.shape[2]：图像的通道数
        temp_crop = img[newy1:newy2, newx1:newx2, :]

        # print(temp_crop.shape)

        # temp_crop = img[det[i, 1]:det[i, 3], det[i, 0]:det[i, 2], :]
        aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        crop.append(aligned)

    # np.stack 将crop由一维list变为二维
    crop_image = np.stack(crop)
    return 1, det_temp, crop_image


def detect_face(scale=2):
    timer = 0
    rtscap = RTSCapture.RTSCapture.create(0)
    rtscap.start_read()  # 启动子线程并改变 read_latest_frame 的指向
    rtscap.set(3, 640 // scale)
    rtscap.set(4, 480 // scale)
    while rtscap.isStarted():
        timer += 1
        start_time = time()
        # Capture frame-by-frame
        ret, frame = rtscap.read_latest_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("esc break...")
            break
        if not ret:
            continue
        resize_img = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
        mark, bounding_box, crop_image = load_and_align_data(resize_img, face_size, 44)
        if (mark):
            face_imgs = np.empty((len(bounding_box), face_size, face_size, 3))
            for i in range(len(bounding_box)):
                face_img = crop_image[i]
                # print(face_img.shape)
                face_imgs[i, :, :, :] = face_img
                # predict ages and genders of the detected faces
            results = model.predict(face_imgs)
            # print(results)
            # print(len(results))
            predicted_genders = results[0]
            # print(len(predicted_genders))
            for j in range(len(predicted_genders)):
                x1 = int(bounding_box[j, 0])
                y1 = int(bounding_box[j, 1])
                x2 = int(bounding_box[j, 2])
                y2 = int(bounding_box[j, 3])
                # print(predicted_genders)
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                # print(predicted_genders)
                label = "{}, {}".format(int(predicted_ages[j]),
                                        "F" if predicted_genders[j][0] > 0.5 else "M")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1,
                            (255, 255, 255),
                            thickness=1,
                            lineType=2)
                print(label)
                # draw_label(frame, (x1, y1), label)
                # cv2.imshow('face_img', face_img)
        cv2.imshow('Keras Faces', frame)
        timer += 1
        end_time = time()
        t = end_time - start_time
        # print(1 // t)
        print(str(int(t * 1000)) + 'ms')
    # When everything is done, release the capture
    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()


def main():
    detect_face(1)


if __name__ == "__main__":
    main()
