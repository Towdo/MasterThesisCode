# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
from scipy.io import loadmat
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
from collections import Counter
from douglaspeucker import DouglasPeucker
from utils import rasterize_strokes_vectorized


# https://github.com/pinakinathc/fscoco/blob/main/src/sbir_baseline/dataloader.py
class FSCOCOLoader(torch.utils.data.Dataset):
    def __init__(self, mode = "normal", max_strokes = None, max_length = None, get_annotations = False, remove_not_annotated = True, do_normalise = True, do_simplify = False, num_examples = None):
        self.mode = mode

        root_dir = os.path.expanduser("~/data/fscoco/")
        vg_annotation_path = os.path.expanduser("~/data/vg/annotations/")

        self.all_image_files = glob.glob(os.path.join(
            root_dir, 'images', '*', '*.jpg'))

        self.ids = sorted([os.path.split(idx)[-1][:-4]
            for idx in glob.glob(os.path.join(
                root_dir, 'vector_sketches', '*', '*.npy'))])

        # Get coco data
        if get_annotations:
            with open(vg_annotation_path + "image_data.json", "r") as f:        # Needed for coco id
                vg_image_data = json.load(f)

            with open(vg_annotation_path + "objects.json", "r") as f:           # Needed for object data
                vg_objects = json.load(f)

            self.objects_coco = dict()
            assert len(vg_image_data) == len(vg_objects)
            for i in tqdm(range(len(vg_image_data)), "Loading annotations", disable=True):
                image_data = vg_image_data[i]
                objects_data = vg_objects[i]
                assert image_data["image_id"] == objects_data["image_id"]
                if image_data["coco_id"] is not None:
                    self.objects_coco[int(image_data["coco_id"])] = Counter([o["names"][0] for o in objects_data["objects"]])

            del vg_image_data
            del vg_objects

            if remove_not_annotated:
                self.ids = [id for id in self.ids if int(id) in self.objects_coco]
                print("Ids remaining", len(self.ids))
            #FIXME:
            self.objects_coco = [list(self.objects_coco[int(id)].keys()) if int(id) in self.objects_coco else set() for id in self.ids]

            max_objects = max([len(objects) for objects in self.objects_coco])

            self.objects_coco = [objects + [""] * (max_objects - len(objects)) for objects in self.objects_coco]

            assert len(self.ids) == len(self.objects_coco)

        ######

        if num_examples is not None:
            self.ids = self.ids[:num_examples]

        self.vector_sketches = [glob.glob(os.path.join(root_dir, 'vector_sketches', '*', '%s.npy'%id))[0] for id in tqdm(self.ids, "Getting paths", disable=True)]
        self.vector_sketches = [np.load(path) for path in tqdm(self.vector_sketches, "Loading sketches", disable=True)]

        if do_simplify:
            self.vector_sketches = [DouglasPeucker(sketch, 2) for sketch in tqdm(self.vector_sketches, "Simplifying", disable=True)]


        def split(input):
            # input[0, 2] = 1 #FSCoco drawings start with 0
            indices = np.where(input[:, 2])[0] + 1
            return np.split(input, indices)

        self.vector_sketches = [split(sketch) for sketch in tqdm(self.vector_sketches, "Splitting sketches", disable=True)]

        #FIXME: Remove this?
        # Remove small strokes
        self.vector_sketches = [[stroke for stroke in sketch if len(stroke) > 1] for sketch in self.vector_sketches]

        if max_length is None:
            max_length = max([max([len(stroke) for stroke in sketch]) for sketch in self.vector_sketches])
        else:
            def split_with_max_size(arr, max_size):
                return [arr[i:i + max_size] for i in range(0, len(arr) - 1, max_size - 1)]
            self.vector_sketches = [sum([split_with_max_size(stroke, max_length) for stroke in sketch], []) for sketch in self.vector_sketches]
        #print(self.vector_sketches[0])
        print("Max length", max_length)#, max([max([len(stroke) for stroke in sketch]) for sketch in self.vector_sketches]))

        if max_strokes is None:
            max_strokes = max([len(sketch) for sketch in self.vector_sketches])
        print("Max strokes", max_strokes)

        self.max_strokes = max_strokes


        self.Nmax = max_length

        def to_5_point(input):
            output = [np.zeros((max_length, 5), np.float32) for _ in range(len(input))]
            values = []         # For normalisation
            lens = [0 for _ in range(len(input))]
            for i, stroke in enumerate(input):
                length = min(len(stroke), max_length)
                lens[i] = length
                output[i][:length, :2] = stroke[:length, :2]
                output[i][1:, :2] = output[i][1:, :2] - output[i][:-1, :2]
                for j in range(length):
                    values.append(output[i][j, 0])
                    values.append(output[i][j, 1])
                output[i][:length, 2] = 1 - stroke[:length, 2]
                output[i][:length, 3] = stroke[:length, 2]
                output[i][length:, 4] = 1
                output[i][length:, :2] = 0  #Zero values

            if do_normalise:                        # Normalise std of actual coordinates
                std = 255#np.max(np.array(values))  #FIXME!!!
                for i in range(len(input)):
                    output[i][:, :2] /= std
            return output, lens
        
        # self.vector_sketches = [to_5_point(sketch) for sketch in tqdm(self.vector_sketches, "To 5 point", disable=True)]
        vector_sketches = []
        lens = []
        for sketch in self.vector_sketches:
            five_point_sketch, length = to_5_point(sketch)
            vector_sketches.append(five_point_sketch)
            lens.append(length)
        self.vector_sketches = vector_sketches
        self.lens = lens
        
        if self.mode == "normal" or self.mode == "rasterize":
            def batch_strokes(sketch, lengths):
                num_strokes = len(sketch)
                output = np.zeros((max_strokes, max_length, 5), np.float32)
                lens = np.zeros((max_strokes), int)
                for i, stroke in enumerate(sketch):
                    length = len(stroke)
                    lens[i] = lengths[i]
                    output[i, :length] = stroke
                output[num_strokes:, :, 4] = 1
                return output, lens
            
            vector_sketches = []
            lens = []
            for i in range(len(self.vector_sketches)):
                batched_sketches, length = batch_strokes(self.vector_sketches[i], self.lens[i])
                vector_sketches.append(batched_sketches)
                lens.append(length)
            print("Prepack")
            self.vector_sketches = torch.from_numpy(np.array(vector_sketches)).contiguous()
            self.stroke_lengths = torch.from_numpy(np.array(lens)).contiguous()
            print("Postpack")

        if self.mode == "rasterize":
            b = len(self.vector_sketches)
            raster_sketches = torch.zeros((b, 256, 256))
            for i in tqdm(range(b)):
                raster_sketches[i] = rasterize_strokes_vectorized(torch.cumsum(self.vector_sketches[i], dim = 1).unsqueeze(0))

            self.raster_sketches = raster_sketches

                

            # self.vector_sketches = [batch_strokes(sketch) for sketch in tqdm(self.vector_sketches, "Batching strokes", disable=True)]

        if self.mode == "pretrain":
            self.strokes = [stroke for sketch in self.vector_sketches for stroke in sketch]
            self.stroke_lengths = [length for sketch_lengths in self.lens for length in sketch_lengths]
            del self.vector_sketches
            

    def __len__(self):
        if self.mode == "normal" or self.mode == "rasterize":
            return len(self.ids)
        if self.mode == "pretrain":
            return len(self.strokes)

    def __getitem__(self, index):
        if self.mode == "normal":
            file_id = self.ids[index]

            return int(file_id), self.vector_sketches[index], self.stroke_lengths[index], self.objects_coco[index]
        
        if self.mode == "rasterize":
            file_id = self.ids[index]

            return int(file_id), self.vector_sketches[index], self.stroke_lengths[index], self.raster_sketches[index], self.objects_coco[index]
        
        if self.mode == "pretrain":
            return self.strokes[index], self.stroke_lengths[index]
    
    
if __name__ == "__main__":

    # data = FSCOCOLoader(opt(), mode = "normal", get_annotations = True, do_normalise = True, do_simplify = True)
    # id, sequence, _ = data[0]

    # import matplotlib.pyplot as plt
    # import PIL

    # def make_image(strokes, epoch, name='_output_'):
    #     """plot drawing with separated strokes"""
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111)
    #     for s in strokes:
    #         stroke = torch.cumsum(s[:, :2], dim = 0)
    #         plt.plot(stroke[:,0],-stroke[:,1])
    #     canvas = plt.get_current_fig_manager().canvas
    #     canvas.draw()
    #     plt.show()

    # make_image(sequence, 0)
    # print(id)

    data = FSCOCOLoader(opt(), mode = "pretrain", do_normalise = True, do_simplify = True)