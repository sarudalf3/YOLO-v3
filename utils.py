import cv2 as cv
import numpy as np
import colorsys
import random

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(1011)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

def post_process(img, outputs, thres, conf, COCO_names, coltext):
	"""Return the post processed image in yolo v3.
		
	Parameters
    ----------
    feats : image to use
	outputs: output of YOLO model
	thres: threshold of object probability
	conf: threshold to IoU
	COCO_names: COCO object names list
	coltext: color of output text in RGB code (0,0,0)
	"""
	H, W = img.shape[:2]
	
	boxes = []
	confidences = []
	classIDs = []

	for output in outputs:
		scores = output[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence > thres:
			x, y, w, h = output[:4] * np.array([W, H, W, H])
			p0 = int(x - w//2), int(y - h//2)
			p1 = int(x + w//2), int(y + h//2)
			boxes.append([*p0, int(w), int(h)])
			confidences.append(float(confidence))
			classIDs.append(classID)

	indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
	colors = get_colors_for_classes(len(COCO_names))

	if len(indices) > 0:
		for i in indices.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in colors[classIDs[i]]]
			cv.rectangle(img, (x, y), (x + w, y + h), color, 5)
            #text = "{}: {:.3f}".format(classes[classIDs[i]], confidences[i])
			text = "{}".format(COCO_names[classIDs[i]])
			cv.rectangle(img, (x, y), (x + w, y - 30), color, -1)
			cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_DUPLEX , 1.4, coltext, 2)
	return img
	


