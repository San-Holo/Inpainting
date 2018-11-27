# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:34:58 2018

@author: admin
"""

from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import hsv_to_rgb
import numpy as np
import numpy.random as r

class Processor:
    def __init__(self, h, filename):
        assert(h % 2 == 1)
        self.h = h
        self.load_img(filename)
        self.optimizer = linear_model.Lasso(alpha = 0.1)
        
    def load_img(self, filename):
        self.img = rgb_to_hsv((plt.imread(filename)[ : , : , : 3 ] / 255)) - 0.5
        
    def patch_as_vector(self, patch):
        print(patch)
        h, w, _ = patch.shape
        return np.reshape(patch, (h * w, 3))
    
    def vector_as_patch(self, vec):
        return np.reshape(vec, (self.h, self.h, 3))
    
    def create_void_patch(self):
        return np.full((self.h, self.h, 3), None)
    
    def replace_patch(self, i, j, patch):
        h_half = int(self.h / 2)
        self.img[i - h_half : i + h_half + 1, j - h_half : j + h_half + 1] = patch
        
    def noise_img(self, prc=0.2):
        nimg = self.img.copy().reshape(-1,3)
        height,width = self.img.shape[:2]
        nb =int(prc*height*width)
        nimg[np.random.randint(0,height*width,nb),:]=None
        self.img = nimg.reshape(height, width, 3)
            
    def retrieve_dico(self, stepping):
        complete = []
        h_half = int(self.h / 2)

        h, w, _ = self.img.shape
        
        h_ratio = int(h / stepping)
        w_ratio = int(w / stepping) 
        
        for i in range(w_ratio):
            for j in range(h_ratio):
                patch = self.get_patch(i * stepping + h_half, j * stepping + h_half)
                if patch.shape[0] != self.h or patch.shape[1] != self.h:
                    continue
                if not np.any(np.isnan(patch)):
                    complete.append(patch)
                    
        return complete
    
    def retrieve_nearby_dico(self, pos, radius):
        complete = []
        h_half = int(self.h / 2)
        
        for i in range(pos[0] - radius, pos[0] + radius + 1):
            for j in range(pos[1] - radius, pos[1] + radius + 1):
                patch = self.get_patch(i + h_half, j + h_half)
                if patch.shape[0] != self.h or patch.shape[1] != self.h:
                    continue
                if not np.any(np.isnan(patch)):
                    complete.append(patch)
                    
        return complete

        

    def show_img(self):
        #print(self.img)
        #print((hsv_to_rgb(self.img + 0.5)))
        im = self.img.copy()
        im[np.isnan(self.img)] = -0.5
        plt.figure()
        plt.imshow((hsv_to_rgb(im + 0.5) * 255).astype(np.uint8))

        
    def get_patch(self, i, j):
        h_half = int(self.h / 2)
        #print(i, j)
        return self.img[i - h_half : i + h_half + 1, j - h_half : j + h_half + 1]
    
    def remove_patch(self, i, j):
        self.replace_patch(i, j, self.create_void_patch())
        
    def remove_region(self, i, j, region_size):
        r_half = int(region_size / 2)
        print(i, j)
        self.img[i - r_half : i + r_half + 1, j - r_half : j + r_half + 1] = None
        
    def remove_random_region(self, region_size):
        h, w, _ = self.img.shape
        self.remove_region(int(r.random() * w), int(r.random() * h), region_size)
        
    def approximate_patch(self, i, j, stepping, radius = 0):
        h_half = int(self.h / 2)
        h, w, _ = self.img.shape


        patch = self.get_patch(i + h_half, j + h_half)
        if np.any(np.isnan(patch)):
            nearby_dic = self.retrieve_nearby_dico((i, j), radius)
            if len(nearby_dic) > 0:
                dic = nearby_dic
            else:
                dic = self.retrieve_dico(stepping)

            y = []
            x = []
            print("frame : ", i, j)

            expressed_pixels = np.logical_not(np.isnan(patch))
            expressed_patch = patch[expressed_pixels]

            if expressed_patch.size == 0:
                return
            for p in expressed_patch:
                y.append(p)
            for elem in dic:
                new_x = elem[expressed_pixels]
                x.append(new_x)
                    
            model = linear_model.LassoCV(max_iter=100000)
            model.fit(np.swapaxes(np.array(x), 0, 1), np.array(y))
            c = model.coef_                    
                    
            for ix in range(self.h):
                for jx in range(self.h):
                    if np.any(expressed_pixels[ix, jx] == False):
                        new_pixel = [0, 0, 0]
   
                        for k in range(len(dic)):
                            new_pixel[0] += c[k] * dic[k][ix][jx][0]
                            new_pixel[1] += c[k] * dic[k][ix][jx][1]
                            new_pixel[2] += c[k] * dic[k][ix][jx][2]
                        print((hsv_to_rgb(np.array(new_pixel)) + 0.5) * 255)
                        self.img[ix + i][jx + j] = np.array(new_pixel)
                    
                                    
        
    def process(self, stepping, radius = 0):
        h, w, _ = self.img.shape
        
        b = 0
        max_x = h - self.h - b
        max_y = w - self.h - b
        did_change = True
        while did_change:
            did_change = False
            for i in range(b, max_x + 1):
                self.approximate_patch(i, b, stepping, radius)
                did_change = True

            for j in range(b, max_y + 1):
                self.approximate_patch(b, j, stepping, radius)
                did_change = True

            for j in range(b, max_y + 1):
                self.approximate_patch(max_x, j, stepping, radius)
                did_change = True

            for i in range(b, max_x + 1):
                self.approximate_patch(i, max_y, stepping, radius)
                did_change = True
            b += 1
            max_x = h - self.h - b
            max_y = w - self.h - b

                                    

np.set_printoptions(threshold=np.nan)        
pr = Processor(3, "tiles.jpg")
pr.show_img()
pr.remove_random_region(10)
pr.remove_random_region(10)
pr.remove_random_region(10)

#pr.noise_img(0.05)
pr.show_img()
#pr.show_img()
pr.process(3, 20)
pr.show_img()