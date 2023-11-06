from collections import Counter
from ultralytics import YOLO
import cvzone
import os
import subprocess
import cv2
from random import randint
color_dict =  {}
model =  YOLO("yolov8s.pt")

def calculate_iou(box1, box2):

      # Extract coordinates of the two boxes
      x1_box1, y1_box1, x2_box1, y2_box1 = box1
      x1_box2, y1_box2, x2_box2, y2_box2 = box2

      # Calculate the coordinates of the intersection
      x1_inter = max(x1_box1, x1_box2)
      y1_inter = max(y1_box1, y1_box2)
      x2_inter = min(x2_box1, x2_box2)
      y2_inter = min(y2_box1, y2_box2)

      # Calculate the area of intersection
      width_inter = max(0, x2_inter - x1_inter)
      height_inter = max(0, y2_inter - y1_inter)
      area_inter = width_inter * height_inter

      # Calculate the area of each bounding box
      area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
      area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

      # Calculate IoU
      iou = area_inter / (area_box1 + area_box2 - area_inter)

      return iou

def color_provider(id):
    """
    Generate a random color for object tracking based on the provided ID.
    """
    color = [randint(0, 256) for _ in range(3)]
    if id in color_dict:
        return color_dict[id]
    else:
        if color in color_dict.values():
            while True:
                color = [randint(0, 256) for _ in range(3)]
                if color not in color_dict.values():
                    break
        color_dict[id] = color
    return color_dict[id]


def convert_video(input_video,output_video):
    if not os.path.exists(output_video):
        command = f"ffmpeg -i {input_video} -vcodec libx264 {output_video}"
        try:
            subprocess.run(command, shell=True, check=True)
            print("Video conversion completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Video conversion failed with error: {e}")
    else:print(f"OUTPUT FOUND : {output_video}")


def combine_all(input_video_path,out_video_path,out_display_video_path,window_size,iou_ther):
# Open the input video
  cap = cv2.VideoCapture(input_video_path)

  # Define the sliding window size
  window_size = (window_size,window_size)

  # Create a VideoWriter for the output video
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  writer = cv2.VideoWriter(out_video_path, fourcc, 25, (output_width, output_height))
  label = ""
  frame_count = 0
  x, y = 0, 0  # Initialize the window position

  while True:
      success, frame = cap.read()

      if not success:
          break

      frame_copy = frame.copy()  # Create a copy of the frame for drawing

      if frame_count % 30 == 0:
          label = ""
          # Extract the window frame
          x1, y1 = x, y
          x2, y2 = x + window_size[0], y + window_size[1]
          window_frame = frame[y1:y2, x1:x2]

          # Process the window frame here
          # You can apply your object counting logic on the 'window_frame'

          # Draw the sliding window on the main frame
          

          # Move the window horizontally
          x += window_size[0]
          if x + window_size[0] > output_width:
              x = 0
              y += window_size[1]
              if y + window_size[1] > output_height:
                  y = 0
      cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
      result =  model.track(frame,persist=True,verbose=False)[0]
      bboxes =  result.boxes.xyxy.int().tolist()
      cls =  result.boxes.cls.int().tolist()
      ids =  result.boxes.id.int().tolist()
      frame_count += 1
      # filter_bboxes=list(filter(lambda bb:calculate_iou(bb,(x1, y1,x2, y2))>0,bboxes))
      bb_cls = [(bb,cl,id)for cl,bb,id in zip(cls,bboxes,ids) if calculate_iou((x1, y1,x2, y2),bb)>iou_ther ]
      for  bb,cl,id in bb_cls:
        b_x1,b_y1,b_x2,b_y2 =  bb
        cv2.rectangle(frame_copy,(b_x1,b_y1),(b_x2,b_y2),color_provider(id),2)
      counter = Counter([c[1] for c in bb_cls])
      count_dict = {model.names[cl]:count for cl,count in counter.items()}
      if True:
        label = ""
        for cl_name,count in count_dict.items():
          label = label + f"{count} {cl_name}--"
          x1_t,y1_t = ((x1+x2)//2,(y1+y2)//2) if y1-30<y1 else (x1,y1-30)
      # cv2.putText(frame_copy,label[:-3],(x1_t,y1_t),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
          frame_copy,_ = cvzone.putTextRect(frame_copy,label[:-2],(x1_t,y1_t),2,2,offset=1)
      # Write the frame with sliding window to the output video
      writer.write(frame_copy)

  # Release video objects
  writer.release()
  cap.release()
  convert_video(out_video_path,out_display_video_path)
