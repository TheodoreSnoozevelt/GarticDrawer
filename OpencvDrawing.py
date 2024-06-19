import cv2
from cv2.typing import MatLike

import Gartic

# From https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
def draw_shape(img: MatLike, shape: Gartic.ToolShape, thicknessScale: float) -> None:
    overlay = img.copy()
    match shape.tool:
        case Gartic.PEN:
            cv2.line(
                overlay,
                (int(shape.a.x), int(shape.a.y)),
                (int(shape.b.x), int(shape.b.y)),
                Gartic.colors[shape.colorIndex],
                max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * thicknessScale), 1
                ),
                lineType=cv2.LINE_AA,
            )

        case Gartic.ELLIPSE_HOLLOW | Gartic.ELLIPSE:
            centerx = int((shape.a.x + shape.b.x) / 2)
            centery = int((shape.a.y + shape.b.y) / 2)
            center_coordinates = (centerx, centery)
            sizex = int(abs(shape.a.x - shape.b.x) / 2)
            sizey = int(abs(shape.a.y - shape.b.y) / 2)
            axes_lengths = (sizex, sizey)
            color = Gartic.colors[shape.colorIndex]

            if shape.tool == Gartic.ELLIPSE_HOLLOW:
                thickness = max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * thicknessScale), 1
                )
            else:
                thickness = -1

            cv2.ellipse(
                overlay,
                center_coordinates,
                axes_lengths,
                0,
                0,
                360,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        case Gartic.RECT_HOLLOW | Gartic.RECT:
            if shape.tool == Gartic.RECT_HOLLOW:
                thickness = max(
                    int(Gartic.thicknesses[shape.thicknessIndex] * thicknessScale), 1
                )
            else:
                thickness = -1

            cv2.rectangle(
                overlay,
                (int(shape.a.x), int(shape.a.y)),
                (int(shape.b.x), int(shape.b.y)),
                Gartic.colors[shape.colorIndex],
                thickness,
                lineType=cv2.LINE_AA,
            )

    cv2.addWeighted(
        overlay,
        Gartic.opacities[shape.opacityIndex],
        img,
        1 - Gartic.opacities[shape.opacityIndex],
        0,
        img,
    )
    del overlay
