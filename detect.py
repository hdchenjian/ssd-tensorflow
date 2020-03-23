import tensorflow as tf
import numpy as np
import cv2

from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps, get_preset_by_name
from utils import draw_box

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph


def main():
    model_file = 'model/ssd300_vgg16_short.pb'
    graph = load_graph(model_file)

    with tf.Session(graph = graph) as sess:
        image_input = sess.graph.get_tensor_by_name('import/define_input/image_input:0')
        result = sess.graph.get_tensor_by_name("import/result/result:0" )

        image_path = 'demo/test.jpg'
        img = cv2.imread(image_path)
        img = np.float32(img)
        img = cv2.resize(img, (300, 300))
        img = np.expand_dims(img, axis=0)
        print('image_input', image_input)
        print('img', type(img), img.shape, img[0][1][1])
        enc_boxes = sess.run(result, feed_dict={image_input: img})
        print('enc_boxes', type(enc_boxes), len(enc_boxes), type(enc_boxes[0]), enc_boxes[0].shape)
        print('detect_result_[0][0]', enc_boxes[0][0])

        lid2name = {0: 'Aeroplane', 1: 'Bicycle', 2: 'Bird', 3: 'Boat', 4: 'Bottle', 5: 'Bus', 6: 'Car',
                    7: 'Cat', 8: 'Chair', 9: 'Cow', 10: 'Diningtable', 11: 'Dog', 12: 'Horse', 13: 'Motorbike',
                    14: 'Person', 15: 'Pottedplant', 16: 'Sheep', 17: 'Sofa', 18: 'Train', 19: 'Tvmonitor'}
        preset = get_preset_by_name('vgg300')
        anchors = get_anchors_for_preset(preset)
        print('anchors', type(anchors))
        boxes = decode_boxes(enc_boxes[0], anchors, 0.5, lid2name, None)
        boxes = suppress_overlaps(boxes)[:200]
        print('boxes', boxes)

        img = cv2.imread(image_path)
        for box in boxes:
            color = (31, 119, 180)
            draw_box(img, box[1], color)

            box_data = '{} {} {} {} {} {}\n'.format(box[1].label, box[1].labelid,
                                                    box[1].center.x, box[1].center.y, box[1].size.w, box[1].size.h)
            print('box_data', box_data)
        cv2.imwrite(image_path + '_out.jpg', img)

if __name__ == '__main__':
    main()
