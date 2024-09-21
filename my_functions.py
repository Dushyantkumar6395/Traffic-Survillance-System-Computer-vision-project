
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
from torchvision import transforms
from PIL import Image
import time



yolov5_weight_file = 'rider_helmet_number_small.pt' # ... may need full path
helmet_classifier_weight = 'helment_no_helmet98.6.pth'
conf_set=0.35 
frame_size=(800, 480) 
head_classification_threshold= 3.0 # make this value lower if want to detect non helmet more aggresively;


# Some list of shape yolov5 except
# 1824, 1376 
# 1024, 576 # cs=4.1
# 928, 544
# 800, 480 # cs=3.9

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True 
names = model.module.names if hasattr(model, 'module') else model.names



### Image classification
# labels = ['helmet', 'no helmet']
model2 = torch.load(helmet_classifier_weight, map_location=device)  # ... may need full path
model2.eval()


transform = transforms.Compose([
			transforms.Resize(144),
			# transforms.CenterCrop(142),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])
		  ]) 


def img_classify(frame):
	# print('Head size: ',frame.shape[:-1])

	if frame.shape[0]<46 : # skiping small size heads <----------------  you can adjust this value
		return [None, 0]

	frame = transform(Image.fromarray(frame))
	frame = frame.unsqueeze(0)
	prediction = model2(frame)
	result_idx = torch.argmax(prediction).item()
	prediction_conf = sorted(prediction[0]) 

	cs = (prediction_conf[-1]-prediction_conf[-2]).item() # confident score
	# print(cs) 
	# provide a threshold value of classification prediction as cs
	if cs > head_classification_threshold: #< --- Classification confident score. Need to adjust, this value
		return [True, cs] if result_idx == 0 else [False, cs]
	else:
		return [None, cs]




def object_detection(frame):
	img = torch.from_numpy(frame) # Convert into a numpy array
	img = img.permute(2, 0, 1).float().to(device)# Reorders the dimensions of the tensor from (height, width, channels) to (channels, height, width) using permute(2, 0, 1),
	img /= 255.0  # Normalize the image tensor
	if img.ndimension() == 3: #Checks if the image tensor has 3 dimensions
		img = img.unsqueeze(0)# changing the tensor’s shape from (channels, height, width) to (1, channels, height, width)

	pred = model(img, augment=False)[0]#Runs the model on the image tensor to get predictions. The augment=False argument indicates that no augmentation is applied during inference
	pred = non_max_suppression(pred, conf_set, 0.30) # prediction, conf, iou (Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes based on the confidence threshold (conf_set) and IoU (Intersection over Union) threshold (0.30))

	detection_result = []# Create empty list
	for i, det in enumerate(pred): #contains a list of detections, where each item is a tensor of detections for a particular scale.
		if len(det): #Checks if there are any detections in the current det tensor
			for d in det: # d = (x1, y1, x2, y2, conf, cls)
				x1 = int(d[0].item()) # Extracts and converts the bounding box coordinates (x1, y1, x2, y2)
				y1 = int(d[1].item())
				x2 = int(d[2].item())
				y2 = int(d[3].item())
				conf = round(d[4].item(), 2)
				c = int(d[5].item())
				
				detected_name = names[c]
                #Prints the detection result including the detected class name, confidence score, and bounding box coordinates.
				print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
				detection_result.append([x1, y1, x2, y2, conf, c]) #Appends the detection result (bounding box coordinates, confidence score, and class label) to the detection_result list
				
				frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1) # box(Draws a bounding box on the frame using OpenCV’s rectangle function. The box is drawn from (x1, y1) to (x2, y2) with a color of (255, 0, 0) (blue) and a thickness of 1 pixel.)
				if c!=1: # if it is not head bbox, then write use putText
					frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

	return (frame, detection_result) #Returns a tuple containing the modified frame with bounding boxes and text annotations, and the detection_result list with all the detected bounding boxes, confidence scores, and class labels.

# Calculates the horizontal and vertical distances between the top-left corners of the small_box and the big_box. Specifically:
def inside_box(big_box, small_box):
    # x1 is the difference between the x-coordinate of the top-left corner of the small_box and the x-coordinate of the top-left corner of the big_box.
	x1 = small_box[0] - big_box[0]
    # y1 is the difference between the y-coordinate of the top-left corner of the small_box and the y-coordinate of the top-left corner of the big_box.
	y1 = small_box[1] - big_box[1]
    # Calculates the horizontal and vertical distances between the bottom-right corners of the big_box and the small_box. Specifically
    # # x2 is the difference between the x-coordinate of the bottom-right corner of the big_box and the x-coordinate of the bottom-right corner of the small_box
	x2 = big_box[2] - small_box[2]
    
    # y2 is the difference between the y-coordinate of the bottom-right corner of the big_box and the y-coordinate of the bottom-right corner of the small_box
	y2 = big_box[3] - small_box[3]
	return not bool(min([x1, y1, x2, y2, 0]))

