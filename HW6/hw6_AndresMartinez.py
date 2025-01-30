import numpy as np
import cv2

def get_hist(image):
    # we just get the histogram with 256 bins (255 and 0 so 256 in total)
    hist = np.bincount(image.ravel(),minlength=256)
    hist = hist.astype(float)
    return hist

def get_im_color(filename):
    # this function loads the image and splits it into red green and blue channels
    # it is split in this specific way since opencv loads it as bgr as default
    img = cv2.imread(filename)
    blue, green, red = cv2.split(img)
    return red,green,blue

def otsu_algorithm(image, foreground=0, background=0, invert=False):
    # performs otsu algorithm, we take in the foreground and background for this since we want to be able to do the iterative otsu algorithm
    # and to handle that we need to know the number of pixels that belong to foreground and background of the image that goes in, 
    # also see the iter_otsu function
    # we first create an empty image which will contain the binary mask
    im = np.zeros_like(image).astype(np.uint8)
    pix = np.arange(0,256,1)

    im_hist = get_hist(image)
    # since we only care about the foreground of the previous step, we substract what used to be the background of the previous step
    # and our total will now be the pixels that belonged to the foreground in the previous step
    im_hist[0] -= background
    im_hist /= foreground
    # we define our maximums as 0 to find the new maximums easily, since they are positive and the gray level for the threshold will be either 0 or greater than 0
    sigma_max = 0.0
    level = 0
    # we iterate through our the positions in our histogram
    for idx in range(1,len(pix)):
        # follow the otsu algorithm equations
        w_0 = im_hist[:idx].sum()
        w_1 = im_hist[idx:].sum()
        if w_0 == 0 or w_1 ==0:
            continue
        mu_0 = (im_hist[:idx] * pix[:idx]).sum()/w_0
        mu_1 = (im_hist[idx:] * pix[idx:]).sum()/w_1

        sigma = w_0*w_1*((mu_0 - mu_1)**2)
        if sigma > sigma_max:
            sigma_max = sigma
            level = idx
    im[image >= level] = 1
    im[image < level] = 0
    im = im.astype(bool)
    # this invert here is for the case when the foreground we want is black, so lower gray values than the background
    if invert:
        return np.logical_not(im)
    return im

def iter_otsu(image, iters=0, invert=False):
    # this is the implementation of the iterative otsu algorithm
    img = image.copy()
    # for the first iteration, the "foreground" is the entire image and the number of pixels in the background is 0
    n_fg = (image.shape[0] * image.shape[1])
    n_bg = 0
    # we run it once and get the foreground
    fg = otsu_algorithm(image, n_fg, n_bg, invert=invert)
    # now if iters > 0 we go into this loop
    for iter in range(iters):
        # we need to know the number of pixels that are in the foreground now, since that is our new "image" (we only care about the foreground)
        n_fg = np.count_nonzero(fg)
        # we get the background pixels easily since its just the total number of pixels - the number of pixels in the foreground
        n_bg = (image.shape[0] * image.shape[1]) - n_fg
        # we mask the image to only consider the foreground
        n_img = cv2.bitwise_and(img,img, mask=(fg*255).astype(np.uint8))
        # we use the masked image, the # of foreground and background pixels to run otsu again
        new_fg = otsu_algorithm(n_img, n_fg, n_bg, invert=invert)
        fg = new_fg
    return fg

def texture_maps(image, N=[3,5,7]):
    maps = []
    for i in range(len(N)):
        # we pad the image so the borders are not ignored, we pad it with the mean since we don't want the variance in those corners to be too big,
        # especially since we then normalize all the variance values in the array to be within 0 and 255
        # we pad it with n//2 for each value of n, we only use 3 values of n
        n = N[i]
        pad = n//2
        n_image = cv2.copyMakeBorder(image, pad,pad,pad,pad, cv2.BORDER_CONSTANT, image.mean())
        sigma_m = np.zeros_like(image,dtype=float)
        # iterate through the image and calculate the variance and save it in the variance array
        for j in range(image.shape[0]):
            for i in range(image.shape[1]):
                window = (n_image[j:(j+n),i:(i+n)]).astype(float)
                window -= window.mean()
                var = window.var()
                sigma_m[j,i] = var
        # we normalize and append it to the map list, we set the type as np.uint8 since its going into the histogram for otsu algorithm
        sigma_m = 255*(sigma_m - sigma_m.min())/(sigma_m.max() - sigma_m.min())
        maps.append(sigma_m.astype(np.uint8))
    return maps

def find_contour(binary_mask, window_size=3):
    # we use a 3x3 window size, this is never changed
    # again, we use padding of 3//2, so 1
    # we find the contour by checking which values of the array are part of the foreground (==1)
    # then check if there is a 0 anywhere in the window, if there is, we know this specific position is a part
    # of the border of the foreground, so we set this position to be 255 so we can save this image with cv2.imwrite
    pad = int(window_size//2)
    contours = np.zeros_like(binary_mask)
    n_bmask = cv2.copyMakeBorder(binary_mask, pad,pad,pad,pad, cv2.BORDER_CONSTANT, 0)
    for j in range(binary_mask.shape[0]):
        for i in range(binary_mask.shape[1]):
            if binary_mask[j,i] == 255:
                window = n_bmask[j:(j+window_size), i:(i+window_size)]
                if 0 in window:
                    contours[j,i] = 255
    return contours.astype(np.uint8)
# for the next two we have invert as a list since we treat each channel differently
# invert[0] is for the red channel/first texture map
# invert[1] is for the green channel/second texture map
# invert[2] is for the blue channel/last texture map
def otsu_rgb(filename, iters=[0,0,0], invert=[False,False,False]):
    # this function does all the necessary calls for the rgb otsu
    img_r, img_g, img_b = get_im_color(filename)
    seg_img_r = iter_otsu(img_r, iters=iters[0], invert=invert[0])
    seg_img_g = iter_otsu(img_g, iters=iters[1], invert=invert[1])
    seg_img_b = iter_otsu(img_b, iters=iters[2], invert=invert[2])
    seg_img = ((seg_img_r*seg_img_g*seg_img_b)*255).astype(np.uint8)
    contour = find_contour(seg_img)
    return seg_img, contour, (seg_img_r*255).astype(np.uint8), (seg_img_g*255).astype(np.uint8), (seg_img_b*255).astype(np.uint8)

def otsu_texture(filename, texture_n=[1,2,3], iters=[0,0,0], invert=[False,False,False]):
    # this is just for the texture based otsu algorithm
    n_list = texture_n
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    map1, map2, map3 = texture_maps(img_gray, N=n_list)
    seg_map1 = iter_otsu(map1, iters[0],invert=invert[0])
    seg_map2 = iter_otsu(map2, iters[1],invert=invert[1])
    seg_map3 = iter_otsu(map3, iters[2],invert=invert[2])
    seg_map = ((seg_map1*seg_map2*seg_map3)*255).astype(np.uint8)
    contour = find_contour(seg_map)
    return seg_map, contour, (seg_map1*255).astype(np.uint8), (seg_map2*255).astype(np.uint8), (seg_map3*255).astype(np.uint8)

### TASK 1
# dog
## rgb otsu
dog = "pics/dog_small.jpg"
iterations = [1,0,0]
seg_dog, dog_cont, seg_r, seg_g, seg_b = otsu_rgb(dog, iters=iterations, invert=[True,True,True])
name = "pics/task_1_dog"
cv2.imwrite(name+".jpg",seg_dog)
cv2.imwrite(name+"_contours.jpg",dog_cont)
cv2.imwrite(name+"_red.jpg",seg_r)
cv2.imwrite(name+"_green.jpg",seg_g)
cv2.imwrite(name+"_blue.jpg",seg_b)
### texture otsu
iterations = [1,0,0]
ns = [9,11,13]
seg_dog, dog_cont, seg_n1, seg_n2, seg_n3 = otsu_texture(dog, texture_n=ns, iters=iterations, invert=[True,True,True])
name = "pics/task_1_texture_dog"
cv2.imwrite(name+".jpg",seg_dog)
cv2.imwrite(name+"_contours.jpg",dog_cont)
cv2.imwrite(name+"_n"+str(ns[0])+".jpg",seg_n1)
cv2.imwrite(name+"_n"+str(ns[1])+".jpg",seg_n2)
cv2.imwrite(name+"_n"+str(ns[2])+".jpg",seg_n3)

# flower
## rgb otsu
flower = "pics/flower_small.jpg"
iterations = [0,0,0]
seg_flower, flower_cont, seg_r, seg_g, seg_b = otsu_rgb(flower, iters=iterations)
name = "pics/task_1_flower"
cv2.imwrite(name+".jpg",seg_flower)
cv2.imwrite(name+"_contours.jpg",flower_cont)
cv2.imwrite(name+"_red.jpg",seg_r)
cv2.imwrite(name+"_green.jpg",seg_g)
cv2.imwrite(name+"_blue.jpg",seg_b)

## texture otsu
iterations = [0,0,0]
ns = [17,19,21]
seg_flower, flower_cont, seg_n1, seg_n2, seg_n3 = otsu_texture(flower, texture_n=ns, iters=iterations)
name = "pics/task_1_texture_flower"
cv2.imwrite(name+".jpg",seg_flower)
cv2.imwrite(name+"_contours.jpg",flower_cont)
cv2.imwrite(name+"_n"+str(ns[0])+".jpg",seg_n1)
cv2.imwrite(name+"_n"+str(ns[1])+".jpg",seg_n2)
cv2.imwrite(name+"_n"+str(ns[2])+".jpg",seg_n3)

### TASK TWO

# golden bust
## rgb otsu
bust = "pics/golden_bust.jpg"
iterations = [0,0,0]
seg_bust, bust_cont, seg_r, seg_g, seg_b = otsu_rgb(bust, iters=iterations, invert=[False,False,True])
name = "pics/task_2_bust"
cv2.imwrite(name+".jpg",seg_bust)
cv2.imwrite(name+"_contours.jpg",bust_cont)
cv2.imwrite(name+"_red.jpg",seg_r)
cv2.imwrite(name+"_green.jpg",seg_g)
cv2.imwrite(name+"_blue.jpg",seg_b)
## texture otsu
iterations = [0,0,0]
ns = [3,5,7]
seg_bust, bust_cont, seg_n1, seg_n2, seg_n3 = otsu_texture(bust, texture_n=ns, iters=iterations)
name = "pics/task_2_texture_bust"
cv2.imwrite(name+".jpg",seg_bust)
cv2.imwrite(name+"_contours.jpg",bust_cont)
cv2.imwrite(name+"_n"+str(ns[0])+".jpg",seg_n1)
cv2.imwrite(name+"_n"+str(ns[1])+".jpg",seg_n2)
cv2.imwrite(name+"_n"+str(ns[2])+".jpg",seg_n3)

# dog in the sand
## rgb otsu
beach = "pics/dog_beach.jpg"
iterations = [2,2,2]
seg_beach, beach_cont, seg_r, seg_g, seg_b = otsu_rgb(beach, iters=iterations)
name = "pics/task_2_dogbeach"
cv2.imwrite(name+".jpg",seg_beach)
cv2.imwrite(name+"_contours.jpg",beach_cont)
cv2.imwrite(name+"_red.jpg",seg_r)
cv2.imwrite(name+"_green.jpg",seg_g)
cv2.imwrite(name+"_blue.jpg",seg_b)
## texture otsu
iterations = [0,0,0]
ns = [3,5,7]
seg_beach, beach_cont, seg_n1, seg_n2, seg_n3 = otsu_texture(beach, texture_n=ns, iters=iterations, invert=[True,True,True])
name = "pics/task_2_texture_dogbeach"
cv2.imwrite(name+".jpg",seg_beach)
cv2.imwrite(name+"_contours.jpg",beach_cont)
cv2.imwrite(name+"_n"+str(ns[0])+".jpg",seg_n1)
cv2.imwrite(name+"_n"+str(ns[1])+".jpg",seg_n2)
cv2.imwrite(name+"_n"+str(ns[2])+".jpg",seg_n3)