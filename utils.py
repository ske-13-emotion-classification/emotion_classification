from cv2 import rectangle


def draw_bounding_boxes(frame, faces):
    for (x, y, w, h) in faces:
        rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + w, y + h),
            color=(0, 255, 0),
            thickness=2,
        )
