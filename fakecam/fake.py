#!/usr/bin/env python3

import asyncio
import itertools
import signal
import sys
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict

import aiohttp
import cv2
import numpy as np
import pyfakewebcam
import requests
import requests_unixsocket
import os
import fnmatch
import time
import threading
import copy

from akvcam import AkvCameraWriter
from pyvisgraph.graph import Graph, Point
from pyvisgraph.visible_vertices import visible_vertices

def findFile(pattern, path):
    for root, _, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None

class RealCam:
    def __init__(self, src, frame_width, frame_height, frame_rate):
        self.cam = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self._set_prop(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._set_prop(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._set_prop(cv2.CAP_PROP_FPS, frame_rate)

    def _set_prop(self, prop, value):
        if self.cam.set(prop, value):
            if value == self.cam.get(prop):
                return True

        print("Cannot set camera property {} to {}, used value: {}".format(prop, value, self.cam.get(prop)))
        return False

    def get_frame_width(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_rate(self):
        return int(self.cam.get(cv2.CAP_PROP_FPS))

    def start(self):
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cam.read()
            if grabbed:
                with self.lock:
                    self.frame = frame.copy()

    def read(self):
        with self.lock:
            if self.frame is None:
               return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()


class FakeCam:
    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        scale_factor: float,
        no_background: bool,
        background_blur: int,
        use_foreground: bool,
        hologram: bool,
        silhouette: bool,
        silhouette_scale_factor: float,
        silhouette_offset_x: int,
        silhouette_offset_y: int,
        beam: bool,
        beam_x: int,
        beam_y: int,
        beam_r: int,
        tiling: bool,
        bodypix_url: str,
        socket: str,
        background_image: str,
        foreground_image: str,
        foreground_mask_image: str,
        webcam_path: str,
        v4l2loopback_path: str,
        use_akvcam: bool
    ) -> None:
        self.no_background = no_background
        self.use_foreground = use_foreground
        self.hologram = hologram
        self.silhouette = silhouette
        self.silhouette_scale_factor = silhouette_scale_factor
        self.silhouette_offset_x = silhouette_offset_x
        self.silhouette_offset_y = silhouette_offset_y
        self.beam = beam
        self.beam_x = beam_x
        self.beam_y = beam_y
        self.beam_r = beam_r
        self.tiling = tiling
        self.background_blur = background_blur
        self.background_image = background_image
        self.foreground_image = foreground_image
        self.foreground_mask_image = foreground_mask_image
        self.scale_factor = scale_factor
        self.real_cam = RealCam(webcam_path, width, height, fps)
        # In case the real webcam does not support the requested mode.
        self.width = self.real_cam.get_frame_width()
        self.height = self.real_cam.get_frame_height()
        self.use_akvcam = use_akvcam
        if not use_akvcam:
            self.fake_cam = pyfakewebcam.FakeWebcam(v4l2loopback_path, self.width, self.height)
        else:
            self.fake_cam = AkvCameraWriter(v4l2loopback_path, self.width, self.height)
        self.foreground_mask = None
        self.inverted_foreground_mask = None
        self.session = requests.Session()
        if bodypix_url.startswith('/'):
            print("Looks like you want to use a unix socket")
            # self.session = requests_unixsocket.Session()
            self.bodypix_url = "http+unix:/" + bodypix_url
            self.socket = bodypix_url
            requests_unixsocket.monkeypatch()
        else:
            self.bodypix_url = bodypix_url
            self.socket = ""
            # self.session = requests.Session()
        self.images: Dict[str, Any] = {}
        self.image_lock = asyncio.Lock()

    async def _get_mask(self, frame, session):
        frame = cv2.resize(frame, (0, 0), fx=self.scale_factor,
                           fy=self.scale_factor)
        _, data = cv2.imencode(".png", frame)
        #print("Posting to " + self.bodypix_url)
        async with session.post(
            url=self.bodypix_url, data=data.tostring(),
            headers={"Content-Type": "application/octet-stream"}
        ) as r:
            mask = np.frombuffer(await r.read(), dtype=np.uint8)
            mask = mask.reshape((frame.shape[0], frame.shape[1]))
            mask = cv2.resize(
                mask, (0, 0), fx=1 / self.scale_factor,
                fy=1 / self.scale_factor, interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
            mask = cv2.blur(mask.astype(float), (30, 30))
            return mask

    def shift_image(self, img, dx, dy):
        img = np.roll(img, dy, axis=0)
        img = np.roll(img, dx, axis=1)
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        return img

    def bounding_box(self, mask):
        # mask has zeros and ones so get a bounding box for that mask
        try:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return ymin, ymax, xmin, xmax
        except:
            # whatever error, just return zeros and caller knows there's no box
            return 0, 0, 0, 0

    def blend_transparent(self, face_img, overlay_t_img):
        # Split out the transparency mask from the colour info
        overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
        overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

    async def load_images(self):
        async with self.image_lock:
            self.images: Dict[str, Any] = {}

            background = cv2.imread(self.background_image, cv2.IMREAD_UNCHANGED)
            if background is not None:
                if not self.tiling:
                    background = cv2.resize(background, (self.width, self.height))
                else:
                    sizey, sizex = background.shape[0], background.shape[1]
                    if sizex > self.width and sizey > self.height:
                        background = cv2.resize(background, (self.width, self.height))
                    else:
                        repx = (self.width - 1) // sizex + 1
                        repy = (self.height - 1) // sizey + 1
                        background = np.tile(background,(repy, repx, 1))
                        background = background[0:self.height, 0:self.width]
                background = itertools.repeat(background)
            else:
                background_video = cv2.VideoCapture(self.background_image)
                self.bg_video_fps = background_video.get(cv2.CAP_PROP_FPS)
                # Initiate current fps to background video fps
                self.current_fps = self.bg_video_fps
                def read_frame():
                        ret, frame = background_video.read()
                        if not ret:
                            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = background_video.read()
                            assert ret, 'cannot read frame %r' % self.background_image
                        frame = cv2.resize(frame, (self.width, self.height))
                        return frame
                def next_frame():
                    while True:
                        self.bg_video_adv_rate = round(self.bg_video_fps/self.current_fps)
                        for i in range(self.bg_video_adv_rate):
                            frame = read_frame();
                        yield frame
                background = next_frame()

            self.images["background"] = background

            if self.use_foreground and self.foreground_image is not None:
                foreground = cv2.imread(self.foreground_image, cv2.IMREAD_UNCHANGED)
                self.images["foreground"] = cv2.resize(foreground,
                                                       (self.width, self.height))
                foreground_mask = cv2.imread(self.foreground_mask_image, cv2.IMREAD_UNCHANGED)
                foreground_mask = cv2.normalize(
                    foreground_mask, None, alpha=0, beta=1,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                foreground_mask = cv2.resize(foreground_mask,
                                             (self.width, self.height))
                self.images["foreground_mask"] = cv2.cvtColor(
                    foreground_mask, cv2.COLOR_BGR2GRAY)
                self.images["inverted_foreground_mask"] = 1 - self.images["foreground_mask"]

    def hologram_effect(self, img):
        # add a blue tint
        holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        holo = cv2.convertScaleAbs(holo, 2.0, 10)
        # add a halftone effect
        bandLength, bandGap = 3, 4
        for y in range(holo.shape[0]):
            if y % (bandLength+bandGap) < bandLength:
                holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, self.shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4, self.shift_image(holo.copy(), -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
        return out

    def silhouette_effect(self, img, mask):
        # create masked image from a frame where non-mask part is transparent
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        for c in range(image.shape[2]):
            image[:, :, c] = 0

        image_silhouette = img.copy()
        image_silhouette = cv2.cvtColor(image_silhouette, cv2.COLOR_RGB2RGBA)
        for c in range(image_silhouette.shape[2]):
            if (c == 3):
                image_silhouette[:, :, c] = image_silhouette[:, :, c] * mask
            else:
                image_silhouette[:, :, c] = image_silhouette[:, :, c]

        # scale silhoutte dx/dy
        scaled_silhouette = cv2.resize(image_silhouette, None, fx=self.silhouette_scale_factor, fy=self.silhouette_scale_factor)
        x_offset = self.silhouette_offset_x
        y_offset = self.silhouette_offset_y

        # place into image and clip if needed
        ss_y_clip = scaled_silhouette.shape[0] if scaled_silhouette.shape[0] + y_offset < image.shape[0] else image.shape[0] - y_offset
        ss_x_clip = scaled_silhouette.shape[1] if scaled_silhouette.shape[1] + x_offset < image.shape[1] else image.shape[1] - x_offset
        image[y_offset:y_offset+ss_y_clip, x_offset:x_offset+ss_x_clip] = scaled_silhouette[0:ss_y_clip, 0:ss_x_clip]

        return image

    def beam_effect(self, img, mask):
        # add hologram beam effect
        beam = img.copy()
        ymin, ymax, xmin, xmax = self.bounding_box(mask.astype(int))
        ymin = int(ymin * self.silhouette_scale_factor)
        xmin = int(xmin * self.silhouette_scale_factor)
        ymax = int(ymax * self.silhouette_scale_factor)
        xmax = int(xmax * self.silhouette_scale_factor)
        ymin += self.silhouette_offset_y
        xmin += self.silhouette_offset_x
        ymax += self.silhouette_offset_y
        xmax += self.silhouette_offset_x

        beam_high_x, beam_high_y, beam_low_x, beam_low_y, beam_center_x, beam_center_y = self.calc_beam_targets(img, mask)
        beam_source_top = [self.beam_x, self.beam_y - self.beam_r]
        beam_target_top = [beam_high_x, beam_high_y]
        beam_target_center = [beam_center_x, beam_center_y]
        beam_target_bottom = [beam_low_x, beam_low_y]
        beam_source_bottom = [self.beam_x, self.beam_y + self.beam_r]

        # print(beam_target_top, beam_target_center, beam_target_bottom)
        if (ymin > 0 and ymin > 0 and xmax > 0 and ymax > 0):
            beam_shape = np.array([beam_source_top, beam_target_top, beam_target_center, beam_target_bottom, beam_source_bottom], np.int32)
            beam = cv2.fillPoly(beam, [beam_shape], (255, 246, 0))
            beam = cv2.GaussianBlur(beam, (25, 25), cv2.BORDER_TRANSPARENT)
            alpha = 0.2
            img = cv2.addWeighted(beam, alpha, img, 1 - alpha, 0)

        return img

    def verts_centroid(self, vertexes):
        high_x = 4094
        high_y = 4094
        low_x = 0
        low_y = 0
        for v in vertexes:
            if int(v[0]) < high_x:
                high_x = int(v[0])
            if int(v[0]) > high_x:
                low_x = int(v[0])
            if int(v[1]) < high_y:
                high_y = int(v[1])
            if int(v[1]) > low_y:
                low_y = int(v[1])
        return (high_x + low_x) / 2, (high_y + low_y) / 2

    def calc_beam_targets(self, img, mask):
        high_x = 4094
        high_y = 4094
        low_x = 0
        low_y = 0

        blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for c in range(blank_image.shape[2]):
            blank_image[:, :, c] = 255 * mask

        imgray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (len(contours)) is 0:
            return 0, 0, 0, 0, 0, 0

        # for now blindly assume that main silhouette contour is a last one
        hull = cv2.convexHull(contours[len(contours) - 1])
        ptsa = hull[:, 0, :]
        center_x, center_y = self.verts_centroid(ptsa);

        points = []
        for p in ptsa:
            x = p[0] * self.silhouette_scale_factor + self.silhouette_offset_x
            y = p[1] * self.silhouette_scale_factor + self.silhouette_offset_y
            points.append(Point(x, y))
        graph = Graph([points])
        visible = visible_vertices(Point(self.beam_x, self.beam_y), graph, None, None)

        for v in visible:
            if int(v.y) < high_y:
                high_x = int(v.x)
                high_y = int(v.y)
            if int(v.y) > low_y:
                low_x = int(v.x)
                low_y = int(v.y)

        center_x = int(center_x * self.silhouette_scale_factor + self.silhouette_offset_x)
        center_y = int(center_y * self.silhouette_scale_factor + self.silhouette_offset_y)

        return high_x, high_y, low_x, low_y, center_x, center_y

    async def mask_frame(self, session, frame):
        # fetch the mask with retries (the app needs to warmup and we're lazy)
        # e v e n t u a l l y c o n s i s t e n t
        mask = None
        while mask is None:
            try:
                mask = await self._get_mask(frame, session)
                # print("Mask: {}".format(mask))
            except Exception as e:
                print(f"Mask request failed, retrying: {e}")
                traceback.print_exc()

        foreground_frame = background_frame = frame
        if self.hologram:
            foreground_frame = self.hologram_effect(foreground_frame)

        overlay = None
        if self.silhouette:
            overlay = self.silhouette_effect(foreground_frame, mask)

        background_frame = cv2.blur(frame, (self.background_blur, self.background_blur), cv2.BORDER_DEFAULT)

        # composite the foreground and background
        async with self.image_lock:
            if self.no_background is False:
                background_frame = next(self.images["background"])

            # overlay simple means we just want a background and overlay atop of it
            # so don't try to mask anything
            if overlay is None:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = foreground_frame[:, :, c] * mask + background_frame[:, :, c] * (1 - mask)
            else:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = background_frame[:, :, c]

            if self.use_foreground and self.foreground_image is not None:
                for c in range(frame.shape[2]):
                    frame[:, :, c] = (
                        frame[:, :, c] * self.images["inverted_foreground_mask"]
                        + self.images["foreground"][:, :, c] * self.images["foreground_mask"]
                        )

            if overlay is not None:
                if self.beam:
                    frame = self.beam_effect(frame, mask)
                frame = self.blend_transparent(frame, overlay);

        return frame

    def put_frame(self, frame):
        self.fake_cam.schedule_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self):
        self.real_cam.stop()
        if self.use_akvcam:
            self.fake_cam.__del__()

    async def run(self):
        await self.load_images()
        self.real_cam.start()
        if self.socket != "":
            conn = aiohttp.UnixConnector(path=self.socket)
        else:
            conn = None
        async with aiohttp.ClientSession(connector=conn) as session:
            t0 = time.monotonic()
            print_fps_period = 1
            frame_count = 0
            while True:
                frame = self.real_cam.read()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                frame = await self.mask_frame(session, frame)
                self.put_frame(frame)
                frame_count += 1
                td = time.monotonic() - t0
                if td > print_fps_period:
                    self.current_fps = frame_count / td
                    print("FPS: {:6.2f}".format(self.current_fps), end="\r")
                    frame_count = 0
                    t0 = time.monotonic()

def parse_args():
    parser = ArgumentParser(description="Faking your webcam background under \
                            GNU/Linux. Please make sure your bodypix network \
                            is running. For more information, please refer to: \
                            https://github.com/fangfufu/Linux-Fake-Background-Webcam")
    parser.add_argument("-W", "--width", default=1280, type=int,
                        help="Set real webcam width")
    parser.add_argument("-H", "--height", default=720, type=int,
                        help="Set real webcam height")
    parser.add_argument("-F", "--fps", default=30, type=int,
                        help="Set real webcam FPS")
    parser.add_argument("-S", "--scale-factor", default=0.5, type=float,
                        help="Scale factor of the image sent to BodyPix network")
    parser.add_argument("-B", "--bodypix-url", default="http://127.0.0.1:9000",
                        help="Tensorflow BodyPix URL")
    parser.add_argument("-w", "--webcam-path", default="/dev/video0",
                        help="Set real webcam path")
    parser.add_argument("-v", "--v4l2loopback-path", default="/dev/video2",
                        help="V4l2loopback device path")
    parser.add_argument("--akvcam", action="store_true",
                        help="Use an akvcam device rather than a v4l2loopback device")
    parser.add_argument("-i", "--image-folder", default=".",
                        help="Folder which contains foreground and background images")
    parser.add_argument("-b", "--background-image", default="background.*",
                        help="Background image path, animated background is \
                        supported.")
    parser.add_argument("--tile-background", action="store_true",
                        help="Tile the background image")
    parser.add_argument("--no-background", action="store_true",
                        help="Disable background image, blurry background")
    parser.add_argument("--background-blur", default="25", type=int,
                        help="Set background blur level")
    parser.add_argument("--no-foreground", action="store_true",
                        help="Disable foreground image")
    parser.add_argument("-f", "--foreground-image", default="foreground.*",
                        help="Foreground image path")
    parser.add_argument("-m", "--foreground-mask-image",
                        default="foreground-mask.*",
                        help="Foreground mask image path")
    parser.add_argument("--hologram", action="store_true",
                        help="Add a hologram effect")

    parser.add_argument("--silhouette", action="store_true",
                        help="Add a silhouette effect")
    parser.add_argument("--silhouette-scale-factor", default=0.4, type=float,
                        help="Scale factor of the silhouette")
    parser.add_argument("--silhouette-offset-x", default=610, type=int,
                        help="Set the offset move by x axis")
    parser.add_argument("--silhouette-offset-y", default=375, type=int,
                        help="Set the offset move by y axis")
    parser.add_argument("--beam", default=False, action="store_true",
                        help="Add a beam effect")
    parser.add_argument("--beam-x", default="1235", type=int,
                        help="Beam x")
    parser.add_argument("--beam-y", default="410", type=int,
                        help="Beam y")
    parser.add_argument("--beam-r", default="8", type=int,
                        help="Beam radius")


    return parser.parse_args()


def sigint_handler(loop, cam, signal, frame):
    print("Reloading background / foreground images")
    asyncio.ensure_future(cam.load_images())


def sigquit_handler(loop, cam, signal, frame):
    print("Killing fake cam process")
    cam.stop()
    sys.exit(0)

def getNextOddNumber(number):
    if number % 2 == 0:
        return number + 1
    return number

def main():
    args = parse_args()
    cam = FakeCam(
        fps=args.fps,
        width=args.width,
        height=args.height,
        scale_factor=args.scale_factor,
        no_background=args.no_background,
        background_blur=getNextOddNumber(args.background_blur),
        use_foreground=not args.no_foreground,
        hologram=args.hologram,
        silhouette=args.silhouette,
        silhouette_scale_factor=args.silhouette_scale_factor,
        silhouette_offset_x=args.silhouette_offset_x,
        silhouette_offset_y=args.silhouette_offset_y,
        beam=args.beam,
        beam_x=args.beam_x,
        beam_y=args.beam_y,
        beam_r=args.beam_r,
        tiling=args.tile_background,
        bodypix_url=args.bodypix_url,
        socket="",
        background_image=findFile(args.background_image, args.image_folder),
        foreground_image=findFile(args.foreground_image, args.image_folder),
        foreground_mask_image=findFile(args.foreground_mask_image, args.image_folder),
        webcam_path=args.webcam_path,
        v4l2loopback_path=args.v4l2loopback_path,
        use_akvcam=args.akvcam)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sigint_handler, loop, cam))
    signal.signal(signal.SIGQUIT, partial(sigquit_handler, loop, cam))
    print("Running...")
    print("Please CTRL-C to reload the background / foreground images")
    print("Please CTRL-\ to exit")
    # frames forever
    loop.run_until_complete(cam.run())


if __name__ == "__main__":
    main()
