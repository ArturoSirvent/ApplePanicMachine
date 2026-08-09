"""
Apple Panic Machine — live COCO detector on Raspberry Pi (+ Coral Edge TPU).

Detects objects from the webcam. Apples and phones trigger sound clips via mplayer.

  python script_for_raspy.py              # preview window
  python script_for_raspy.py --headless   # no window (audio / prints only)
  python script_for_raspy.py --no-tpu     # CPU TFLite model (no Coral)

Quit: 'q' in the window, or Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from subprocess import Popen
from threading import Thread

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# VideoStream: threaded webcam reader (Adrian Rosebrock / PyImageSearch)
# https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/


class VideoStream:
    """Webcam stream in a background thread."""

    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def play_sound(path: Path) -> None:
    Popen(["mplayer", str(path)])


def load_interpreter(model_dir: Path, use_tpu: bool) -> Interpreter:
    if use_tpu:
        from tflite_runtime.interpreter import load_delegate

        model = model_dir / "tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
        interpreter = Interpreter(
            model_path=str(model),
            experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
        )
    else:
        model = model_dir / "tf2_ssd_mobilenet_v2_coco17_ptq.tflite"
        interpreter = Interpreter(model_path=str(model))

    interpreter.allocate_tensors()
    return interpreter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apple Panic Machine")
    p.add_argument(
        "--headless",
        action="store_true",
        help="no OpenCV window (for headless Pi)",
    )
    p.add_argument(
        "--no-tpu",
        action="store_true",
        help="run without Coral Edge TPU",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.65,
        help="min detection confidence (default: 0.65)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_dir = root / "modelos"
    data_dir = root / "datos"

    use_tpu = not args.no_tpu
    interpreter = load_interpreter(model_dir, use_tpu)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(model_dir / "coco_labels.txt") as f:
        coco_classes = np.array(f.read().splitlines())

    height, width = input_details[0]["shape"][1], input_details[0]["shape"][2]
    res_w, res_h = 640, 480
    videostream = VideoStream(resolution=(res_w, res_h), framerate=30).start()
    time.sleep(1)

    min_conf = args.conf
    frame_rate_calc = 1.0
    freq = cv2.getTickFrequency()

    # rolling window of recent labels; fire audio if a keyword shows up often enough
    last_detections: list[str] = []
    target_apple = "apple"
    target_phone = "cell phone"
    window_size = 20
    hit_threshold = 2

    print(
        f"APM running (TPU={use_tpu}, headless={args.headless}). "
        "Ctrl+C to stop" + ("" if args.headless else ", or 'q' in the window") + "."
    )

    try:
        while True:
            t1 = cv2.getTickCount()
            frame = videostream.read()
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            scores = interpreter.get_tensor(output_details[0]["index"])[0]
            boxes = interpreter.get_tensor(output_details[1]["index"])[0]
            classes = interpreter.get_tensor(output_details[3]["index"])[0]

            apple_area = None

            for i in range(len(scores)):
                if not (min_conf < scores[i] <= 1.0):
                    continue

                ymin = int(max(1, boxes[i][0] * res_h))
                xmin = int(max(1, boxes[i][1] * res_w))
                ymax = int(min(res_h, boxes[i][2] * res_h))
                xmax = int(min(res_w, boxes[i][3] * res_w))
                rel_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])

                object_name = str(coco_classes[int(classes[i])])
                last_detections.append(object_name)
                if object_name == target_apple:
                    apple_area = rel_area

                if not args.headless:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    label = "%s: %d%%" % (object_name, int(scores[i] * 100))
                    label_size, base_line = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(
                        frame,
                        (xmin, label_ymin - label_size[1] - 10),
                        (xmin + label_size[0], label_ymin + base_line - 10),
                        (255, 255, 255),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )

            if len(last_detections) > window_size:
                last_detections = []
            elif last_detections:
                n_apple = sum(1 for x in last_detections if x == target_apple)
                n_phone = sum(1 for x in last_detections if x == target_phone)

                if n_apple > hit_threshold:
                    # closer apple (bigger box) → bigger freakout
                    if apple_area is not None and apple_area < (0.4 * 0.4):
                        play_sound(data_dir / "oh_no2_crop.mp3")
                        print("oh no")
                    else:
                        play_sound(data_dir / "nogod_crop.mp3")
                        print("NO GOD NO")
                    time.sleep(2 if args.headless else 3)
                    last_detections = []

                if n_phone > hit_threshold:
                    play_sound(data_dir / "okey.mp3")
                    print("okey")
                    time.sleep(2 if args.headless else 3)
                    last_detections = []

            t2 = cv2.getTickCount()
            frame_rate_calc = 1.0 / ((t2 - t1) / freq)
            # keep the rolling window roughly ~1 second of detections
            window_size = max(5, int(frame_rate_calc))

            if not args.headless:
                cv2.putText(
                    frame,
                    "FPS: {0:.2f}".format(frame_rate_calc),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Object detector", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        videostream.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
