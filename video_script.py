import cv2

file_path = './akn.155.086.left.avi'
file_out = '..\data\pred_akn.155.086.left.avi'

detect = lambda x: 0

video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'H264')

out_video = cv2.VideoWriter(file_out, fourcc, fps=fps, frameSize=(640,480))


while video.isOpened():
    ret, frame = video.read()
    detect(frame)
    pred_frame = cv2.imread('predictions.jpg')
    out_video.write(pred_frame)

video.release()
out_video.release()