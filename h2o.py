"""This class implements the model-agnostic, class-agnostic hierarchical occlusion-based explanation 
method called H^2O.
"""

__author__ = ["Luc Etienne Pommé", "Romain Bourqui"]
__copyright__ = "Copyright (C) Univ. de Bordeaux"
__credits__ = ["Luc Etienne Pommé", "Romain Bourqui", "Romain Giot"]

__license__ = "GNU General Public License, version 3"
__version__ = "1.0"
__maintainer__ = "Luc Etienne Pommé"
__email__ = "luc.pomme-cassierou@u-bordeaux.fr"

"""
 This file is part of H2O.

H2O is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Foobar is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>.
"""

import gc
import itertools
import numpy as np

import cmapy
from skimage.segmentation import slic as skslic
from scipy.ndimage.filters import gaussian_filter


class H2O():
    def __init__(self, n=4, min_size=1000, predict_fct=None, nb_classes=0, max_iter=10):
        """
        Description
            Implementation of the model-and-class-agnostic occlusion-based explanation method called H2O. Given a model and an input image, this class provides the methods to compute the explanation as a heatmap. 

        Parameters
            n: the maximum number of superpixels a segmentation algorithm is authorized to produce, in every level of the hierarchy.
            min_size: the number of pixels below which a superpixel is not segmented anymore in the hierarchy
            predict_fct: the function pointer to make a prediction from a model.
            nb_classes: the number of classes in the classification task
            max_iter: the maximum number of iterations in the SLIC algorithm to hierarchically segment the input image, optional

        Returns
            combi_masks: the set of masks computed to alter the input image
            nb_evaluation_per_pixels: an array containing the number of times each pixel appears in a mask
        """
        self.n = n
        self.colors = []
        self.nb_classes = nb_classes
        self.min_size = min_size
        self.max_iter = max_iter
        self.predict_fct = predict_fct

    def compute_combinations(self, n):
        """
        Description
            Given a list of n elements, computes the list of all combinations of k elements, for k in [1, n-1].

        Parameters
            n: the number of elements in the set

        Returns
            combinations: the list of combinations of all k elements, by index
        """
        combinations = []
        for i in range(1, len(n)):
            combinations += (itertools.combinations(n, i))
        return combinations

    def compute_predictions(self, images, model):
        """
        Description
            Computes the predictions on altered images by batch. 

        Parameters
            images: the altered images to predict
            model: the model to explain

        Returns
            predictions: the prediction vectors for each altered image
        """
        nb_imgs = np.shape(images)[0]
        start = 0
        step = 1024
        stop = nb_imgs
        predictions = np.empty((nb_imgs, self.nb_classes))
        while start + step < stop:
            predictions[start:start+step] = self.predict_fct(model, images[start:start+step])
            start += step
        if start != stop:
            predictions[start:stop] = self.predict_fct(model, images[start:stop])
        return predictions

    def get_predictions(self, model, image):
        """
        Description
            Use the prediction function pointer to make a model-agnostic prediction on an image without.

        Parameters
            model: the model to explain
            image: the input image

        Returns
            prediction: the prediction vector for the input image
        """
        # Remove the batch dimension
        return np.asarray(self.predict_fct(model, image)[0])

    def color_strategy(self, image, z):
        """
        Description
            Determines the masking colors strategy.

        Parameters
            image: the input image
            z: the number of splits per channel in the RGB cube restricted to the colors in the input image, to determine the number of masking colors. k subdivisions will generate k^3 colors
        """
        self.colors = []
        if (len(image.shape) == 3 and image.shape[2] == 3):
            maxs = np.amax(image, axis=(0, 1))
            mins = np.amin(image, axis=(0, 1))

            step_r = (maxs[0]-mins[0])/(z-1)
            step_g = (maxs[1]-mins[1])/(z-1)
            step_b = (maxs[2]-mins[2])/(z-1)

            start_r = mins[0]
            start_g = mins[1]
            start_b = mins[2]

            for rstep in range(z):
                for gstep in range(z):
                    for bstep in range(z):
                        self.colors.append((start_r+rstep*step_r, start_g+gstep*step_g, start_b+bstep*step_b))

    def generate_mask_combinations(self, image, slic_sigma, channel):
        """
        Description
            Hierarchically segments the input image into superpixels using the SLIC algorithm and computes a mask for each combination of superpixels (except the two trivial combinations).

        Parameters
            image: the input image
            slic_sigma: the width of the Gaussian smoothing kernel for pre-processing for each dimension of the image in the SLIC algorithm.
            channel: the channel dimension index in the input image

        Returns
            combi_masks: the set of masks computed to alter the input image
            nb_evaluation_per_pixels: an array containing the number of times each pixel appears in a mask
        """
        w = image.shape[0]
        h = image.shape[1]
        convert2lab = len(np.shape(image)) == 3
        queue_masks = np.full((1, w, h), True)
        nb_evaluation_per_pixels = np.zeros((w, h))
        combi_masks = []
        q_ind = 0

        while queue_masks.shape[0] - q_ind > 0:
            mask = queue_masks[q_ind]
            q_ind += 1
            labels = skslic(image=image,
                            n_segments=self.n,
                            compactness=1.,  # ,
                            max_num_iter=self.max_iter,
                            sigma=slic_sigma,
                            spacing=None,
                            convert2lab=convert2lab,
                            enforce_connectivity=False,
                            min_size_factor=0.5,
                            max_size_factor=self.n,
                            slic_zero=False,
                            start_label=0,
                            mask=mask,
                            channel_axis=channel,
                            )
            unique_lbls = list(np.unique(labels))
            if -1 in unique_lbls:
                unique_lbls.pop(unique_lbls.index(-1))
            if len(unique_lbls) >= 2:
                for spx_ind in unique_lbls:
                    spx_mask = labels == spx_ind
                    if np.sum(spx_mask) > self.min_size:
                        spx_mask = np.reshape(spx_mask, (1, spx_mask.shape[0], spx_mask.shape[1]))
                        queue_masks = np.concatenate((queue_masks, spx_mask), axis=0)

                combinations = self.compute_combinations(unique_lbls)
                for combi in combinations:
                    combi_msk = np.full((w, h), False)
                    for num in combi:
                        combi_msk[labels == num] = True
                        nb_evaluation_per_pixels[labels == num] += 1
                    combi_masks.append(combi_msk)
        del queue_masks
        gc.collect()
        return combi_masks, nb_evaluation_per_pixels

    def alter_and_predict_images(self, original_image, combi_masks, model):
        """
        Description
            For each computed mask, alter the input image and computes the deterioration induced on the prediction vector. 

        Parameters
            original_image: the input image
            combi_masks: the set of masks computed to alter the input image
            model: the model to explain

        Returns
            predictions: the array of prediction vectors per altered image
        """
        nb_colors = len(self.colors)
        nb_masks = len(combi_masks)
        predictions = np.empty(shape=(nb_colors, nb_masks, self.nb_classes))
        for icolor, color in enumerate(self.colors):
            masked_images = np.repeat(np.expand_dims(original_image, axis=0), nb_masks, axis=0)
            for imask, mask in enumerate(combi_masks):
                masked_images[imask][mask] = color
            predictions_for_color = self.compute_predictions(masked_images, model)
            predictions[icolor] = predictions_for_color
            del masked_images
        return predictions.reshape((nb_masks*nb_colors, self.nb_classes), order="F")

    def alteration_dist(self, prediction, base_prediction):
        """
        Description
            Computes a pseudo-manhattan distance between the prediction vectors of two images, weighted by the prediction scores of the input image. Only deteriorations are counted in that distance.

        Parameters
            prediction: the prediction vector of an altered image
            base_prediction: the prediction vector of the input image

        Returns
            res: the distance value between the two input vectors
        """
        res = base_prediction - prediction
        res[res < 0] = 0
        return np.sum(res * base_prediction)

    def compute_alteration_scores(self, total_nb_masks, predictions, base_pred):
        """
        Description
            Measures a distance between the base prediction vector and the prediction vector of every altered image.

        Parameters
            total_nb_masks: the total number of masks (nb_masks * nb_colors)
            predictions: the array containing the prediction vectors for each altered image
            base_pred: the prediction vector of the input image

        Returns
            d_pred: the array containing the alteration score for each altered image 
        """
        d_pred = np.zeros(total_nb_masks)
        for i, _ in enumerate(d_pred):
            d_pred[i] = self.alteration_dist(predictions[i], base_pred)
        return d_pred

    def aggregate_alter_scores(self, w, h, combi_masks, alter_scores, nb_clrs, nb_px_per_mask, nb_eval_per_pixels):
        """
        Description
            Aggregates the deterioration scores of all altered images to generate a unique saliency map for the input image. 

        Parameters
            w: the width of the input image
            h: the height of the input image
            combi_masks: the set of masks computed to alter the input image
            alter_scores: the deterioration scores computed for each altered image
            nb_clrs: the number of masking colors 
            nb_px_per_mask: an array containing the number of pixels in each mask
            nb_eval_per_pixels: an array containing the number of times each pixel appears in a mask

        Returns
            saliency_map: the saliency map computed from the aggregation of the saliency maps computed for each altered image
        """
        saliency_map = np.zeros((w, h))
        scores = np.mean((alter_scores).reshape((-1, nb_clrs)), axis=1) / nb_px_per_mask
        for imask, mask in enumerate(combi_masks):
            saliency_map[mask] += scores[imask]
        saliency_map = saliency_map / nb_eval_per_pixels
        return saliency_map

    def h2o_explain_image(self, image, model, channel=-1, subdivision_per_channels=3, slic_sigma=1):
        """
        Description
            Computes a class-agnostic saliency map explanation for the input image, based on the model predictions.

        Parameters
            image: the input image to explain
            model: the model to explain
            channel: the channel dimension index in the input image
            subdivision_per_channels: the number of splits per channel in the RGB cube restricted to the colors in the input image, to determine the number of masking colors. k subdivisions will generate k^3 colors
            slic_sigma: the width of the Gaussian smoothing kernel for pre-processing for each dimension of the image in the SLIC algorithm.

        Returns
            saliency_map: the raw saliency map explaining the prediction vector of the model on the input image.
        """
        base_pred = self.get_predictions(model, image)
        image = image[0]
        w = image.shape[0]
        h = image.shape[1]

        self.color_strategy(image, z=subdivision_per_channels)
        combi_masks, nb_eval_per_pixels = self.generate_mask_combinations(image, slic_sigma, channel)

        nb_colors = len(self.colors)
        total_nb_masks = nb_colors * len(combi_masks)
        predictions = self.alter_and_predict_images(image, combi_masks, model)
        alteration_scores = self.compute_alteration_scores(total_nb_masks, predictions, base_pred)

        nb_px_per_mask = np.sum(combi_masks, axis=(1, 2))
        saliency_map = self.aggregate_alter_scores(w, h, combi_masks, alteration_scores, nb_colors,
                                                   nb_px_per_mask, nb_eval_per_pixels)

        del combi_masks
        del nb_px_per_mask
        del predictions
        gc.collect()
        return saliency_map

    def superimpose_image_heatmap(self, image, heatmap, alpha=0.6, beta=0.4):
        """
        Description
            Computes the weigthed superimposition of the input image and the computed heatmap.

        Parameters
            alpha: Weight of the input image
            beta: Weight of the heatmap

        Returns
            superimposed: the superimposition of the image and the heatmap
        """
        return alpha * image + beta * heatmap

    def threshold_hms(self, saliency_map, k_sigma=1.):
        """
        Description
            Applies a threshold on the input saliency map. All values below $mu + k_sigma * sigma$, where mu and sigma are the mean and standard deviation of the values in the saliency map, are nullified.

        Parameters
            saliency_map: the computed saliency map
            k_sigma: the factor applied to the standard deviation to threshold the saliency map

        Returns
            saliency_map: the thresholded saliency map
        """
        mu_hm = np.mean(saliency_map)
        sigma_hm = np.std(saliency_map)
        th = mu_hm + k_sigma * sigma_hm
        min_hm = np.min(saliency_map)
        saliency_map[saliency_map < th] = min_hm
        return saliency_map

    def normalize(self, array):
        """
        Description
            Normalize the input array with a min-max normalization.

        Parameters
            output_scores: the array containing final scores for each pixel

        Returns
            array: the normalized array
        """
        m = np.min(array)
        M = np.max(array)
        if M-m == 0:
            return array
        return (array - m) / (M - m)

    def final_heatmap(self, image, saliency_map, apply_gaussian=True, sigma=3, threshold=True, sm_save_path=None):
        """
        Description
            Returns a heatmap representation from an image and its saliency map. 

        Parameters
            image: the image to explain
            saliency_map: the computed saliency map
            apply_gaussian: a boolean to indicate whether to apply a gaussian kernel on the heatmap
            sigma: the desired standard deviation of the gaussian kernel (if applied to the heatmap)
            threshold: a boolean to indicate whether to apply a threshold on the heatmap
            sm_save_path: the path to save the saliency map array (before thresholding and blurring)

        Returns
            stack: an image composed of: 
                input image | heatmap | (input_image + heatmap)
        """
        original_img = np.copy(image).astype('uint8')
        original_img = np.squeeze(original_img)

        saliency_map = self.normalize(saliency_map) * 255
        saliency_map = saliency_map.astype('uint8')
        if sm_save_path is not None:
            np.save(sm_save_path, saliency_map)
        if threshold:
            saliency_map = self.threshold_hms(saliency_map)
        if apply_gaussian:
            saliency_map = gaussian_filter(saliency_map, sigma)
        saliency_map = cmapy.colorize(saliency_map, 'turbo', rgb_order=True)

        final_image = self.superimpose_image_heatmap(original_img, saliency_map)
        final_image = np.squeeze(final_image)
        final_image = np.hstack((original_img, saliency_map, final_image))

        return final_image.astype('uint8')
