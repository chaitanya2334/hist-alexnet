# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
import os
from pathlib import Path

import cv2
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from utils import prob


class CAM():
    def __init__(self):
        self.features_blobs = None

    @staticmethod
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 224x224
        size_upsample = (224, 224)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    @staticmethod
    def rescale(mat, r, t):
        # r (rmin, rmax)
        # t (tmin, tmax)
        mat = ((mat - r[0]) / (r[1] - r[0])) * (t[1] - t[0]) + t[0]
        return mat

    def blend(self, fg, bg, alpha):
        fg = fg.astype(float)
        bg = bg.astype(float)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = (alpha.astype(float) / 255)

        alpha = self.rescale(alpha, (0, 1), (0.3, 1))

        # Multiply the foreground with the alpha matte
        fg = fg * alpha[:, :, None]

        # Multiply the background with ( 1 - alpha )
        alpha_1 = 1.0 - alpha
        bg = bg * alpha_1[:, :, None]

        # Add the masked foreground and background.
        outImage = cv2.add(fg, bg)

        return outImage

    @staticmethod
    def create_blank(width, height, rgb_color=(0, 0, 0)):
        """Create new image(numpy array) filled with certain color in RGB"""
        # Create black blank image
        image = np.zeros((height, width, 3), np.uint8)

        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed(rgb_color))
        # Fill image with color
        image[:] = color

        return image

    def gen_cam_saliency(self, image, model, final_conv_name, classes):
        model.eval()

        def hook_feature(module, input, output):
            self.features_blobs = output.data.cpu().numpy()

        model._modules.get(final_conv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(model.parameters())
        last_layer = params[-2].data.cpu().numpy()
        weight_softmax = np.squeeze(last_layer)

        logit = model(image)

        # print(weight_softmax.shape, last_layer.shape, self.features_blobs.shape)

        probs = F.softmax(logit, dim=1).data.squeeze()

        res = []
        # render the CAM and output
        inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        img = inv_normalize(image.squeeze()).data.cpu().numpy() * 255
        img = np.clip(img, 0, 255)
        img = np.array(img, dtype=np.uint8)
        img = np.transpose(img, (1, 2, 0))

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(len(classes)):
            # generate class activation mapping for the top1 prediction
            CAMs = self.returnCAM(self.features_blobs, weight_softmax, [i])

            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_BONE)
            heatmap = 255 - heatmap

            white_bg = self.create_blank(width, height, rgb_color=(0, 100, 0))
            result = self.blend(img, white_bg, CAMs[0])

            res.append((img, result, CAMs[0]))

        return res

    def save_image(self, save_dir, file_name, image):
        dir1 = os.path.join(*Path(file_name).parts[-3:])
        save_path = os.path.join(save_dir, dir1)
        # print(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)

    def visualize(self, data_loader, model, classes, _dir):
        c = 0
        for image, label, image_path in tqdm(data_loader, desc="visualizing", total=np.math.ceil(len(data_loader))):
            model.zero_grad()
            label = label[0]
            image_path = image_path[0]
            image = Variable(image).cuda()
            seq_out = model(image)
            tile_name = "t" + str(c)

            # tracker = SummaryTracker()
            res = self.gen_cam_saliency(image, model, 'features', classes)

            for i, (orig, result, heatmap) in enumerate(res):
                # save image
                save_dir = os.path.join(_dir, "saliency_maps")
                os.makedirs(save_dir, exist_ok=True)
                p = prob(seq_out, i, decimals=4)
                if len(p) == 1:
                    p = p[0]

                for _type, img in [("original", orig), ("blend", result), ("heatmap", heatmap)]:
                    self.save_image(save_dir=os.path.join(save_dir, Path(image_path).stem),
                                    file_name="{0}_{1}_({2})_{3}.png".format(tile_name, classes[i], str(p), _type),
                                    image=img)

            c += 1

            # tracker.print_diff()
            # all_objects = muppy.get_objects()
            # print([(ao, ao.shape) for ao in all_objects if isinstance(ao, np.ndarray)])
