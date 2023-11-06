import streamlit as st
import os
from detector import combine_all

st.title("Zone Object Detector")

out_zip_path = "out.zip"
out_display_video_path = "out_display.mp4"
input_video_path = st.text_input("enter the input video path")
if input_video_path:
  st.write("input video")
  st.video(input_video_path)
  out_video_path = "out_video.mp4"
  window_size = st.text_input("enter the window size in pixels")
  iou_ther = st.text_input("enter the iou ther")

  if all([out_video_path,window_size,iou_ther]):
    combine_all(input_video_path,
    out_video_path,out_display_video_path,int(window_size),float(iou_ther))

  if os.path.isfile(out_display_video_path):
    st.write("output video")
    st.video(out_display_video_path)
    with open(out_display_video_path,"rb") as f:
      st.download_button("download",f.read(),"output_zone.mp4")
