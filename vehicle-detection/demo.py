from utils import *
from darknet import Darknet
import cv2

EXIT_PTS = np.array([
    [[645, 720], [645, 500], [1280, 500], [1280, 720]],
    [[0, 500], [645, 500], [645, 0], [0, 0]]
])
SHAPE = (720, 1280)
base = np.zeros(SHAPE + (3,), dtype='uint8')
exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]
context = {}

def demo(cfgfile, weightfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    elif m.num_classes == 4:
        namesfile = 'data/vec.names'
    else:
        namesfile = 'data/names'
    class_names = load_class_names(namesfile)
 
    use_cuda = 1
    if use_cuda:
        m.cuda()

    vehicleCounter = VehicleCounter(exit_masks=[exit_mask], class_names=class_names, y_weight=2.0)
    visualizer = Visualizer()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/william/project/car_detection2/input.mp4')

    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
#image size 720,1280
    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img, (m.width, m.height))
            bboxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            print('------')
            # draw_img = plot_boxes_cv2(img, bboxes, None, class_names)

            context = {}
            context['frame'] = img
            context['class_names'] = class_names
            context['objects'] = box_convert(bboxes, width=SHAPE[1], height=SHAPE[0])
            context = vehicleCounter(context)


            visualizer(context)
            # cv2.imshow(cfgfile, draw_img)
            # cv2.waitKey(1)
        else:
             print("Unable to read image")
             exit(-1) 

############################################
if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        demo(cfgfile, weightfile)
        #demo('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights')
    else:
        print('Usage:')
        print('    python demo.py cfgfile weightfile')
        print('')
        print('    perform detection on camera')
