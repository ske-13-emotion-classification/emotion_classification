from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
from numpy import add, array
import cv2


def crop_image(image, bounding_box):
    (x, y, width, height) = bounding_box
    return image[y:y + height, x:x + width]


def extract_objects(image, bounding_boxes):
    return [crop_image(image, bounding_box) for bounding_box in bounding_boxes]


def draw_bounding_box(image, bouding_box, color=(0, 255, 0), thickness=2):
    x, y, width, height = bouding_box
    rectangle(
        img=image,
        pt1=(x, y),
        pt2=(x + width, y + height),
        color=color,
        thickness=thickness,
    )


def draw_bounding_boxes(image, bouding_boxes=[], color=(0, 255, 0), thickness=2):
    for box in bouding_boxes:
        draw_bounding_box(
            image=image,
            bouding_box=box,
            color=color,
            thickness=thickness,
        )


def draw_texts(image, texts=[], bounding_boxes=[]):
    for (text, box) in zip(texts, bounding_boxes):
        draw_text(image=image, text=text)


def draw_text(image, text='Text', coordinate=[0, 0], offsets=[1, 1], font_scale=2, color=(255, 0, 0), thickness=2):
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), lineType=cv2.LINE_AA)
