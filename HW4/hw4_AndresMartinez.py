import cv2
import numpy as np

### FUNCTIONS
def grayscale(image):
    # get grayscale
    # this follows the opencv color conversion documentation
    # I know that this is easily done with cvtColor but I wanted to practice broadcasting
    avg = np.array([0.114,0.587,0.299],dtype=float)
    avg = avg.reshape((1,1,3,1))
    img = image[:,:,np.newaxis,:]
    img = img @ avg
    return img[:,:,0,0]

def haar_filter(sig):
    # get both Haar x and Haar y from this
    M = np.ceil(sig*4)
    M = int(M + (M%2)) # the mod 2 ensures that this is an even number since it is 1 when its odd, making it even now
    haar_x = np.ones((M,M))
    haar_x[:,:int(M/2)] *= -1
    haar_y = np.ones((M,M))
    haar_y[:int(M/2),:] *= -1
    return haar_x,haar_y

def normalize(image):
    # turns a grayscale image from [0, 255] range to [0,1]
    # I tested with a different normalization but this works best
    return image/255. #(image - np.min(image))/(np.max(image) - np.min(image))

def harris_corner(image, sig, k=0.05):
    # we start off by normalizing the grayscale image
    graynorm_img = normalize(grayscale(image))
    haar_x, haar_y = haar_filter(sig)
    # convolve with the filters
    dx = cv2.filter2D(graynorm_img,ddepth=-1,kernel=haar_x)
    dy = cv2.filter2D(graynorm_img,ddepth=-1,kernel=haar_y)
    # 5*sigma neighborhood
    M = int(np.ceil(5*sig))
    nh = np.ones((M,M))
    dxdx = cv2.filter2D(dx*dx,ddepth=-1,kernel=nh)
    dxdy = cv2.filter2D(dx*dy,ddepth=-1,kernel=nh)
    dydy = cv2.filter2D(dy*dy,ddepth=-1,kernel=nh)
    #calculate Harris response
    tr_c = dxdx * dydy
    det_c = dxdx*dydy - (dxdy)**2
    M = M*2
    R = det_c - k*(tr_c**2)
    # we threshold with 80% of the maximum response value
    R_threshold = 0.8*R.max()
    corners = []
    # non-max suppression to only keep the max value in a neighborhood
    for i in range(M//2, R.shape[1] - M//2):
        for j in range(M//2, R.shape[0] - M//2):
            window = R[int(j-M/2):int(j+M/2),int(i-M/2):int(i+M/2)]
            if R[j,i]==window.max() and R[j,i] >= R_threshold:
                corners.append([i,j,R[j,i]])
    corners = np.array(corners)
    #we sort them by response magnitude, descending
    corners_sorted = corners[corners[:,2].argsort()[::-1]]
    return corners_sorted[:,:2].astype(int)


def show_corners(image, corners,n_corners):
    # this function is just to plot the first n_corners number of corners in the image
    img = image.copy()
    if len(img.shape) < 3:
        # this part is to deal with any grayscale image since we want the corners to be blue (needs to be rgb image)
        img = img[:,:,np.newaxis].tile(1,1,3)
    for corner in corners[:n_corners]:
        x, y = corner
        img = cv2.circle(img,(x,y),4,(255,0,0),-1)
    return img

def get_neighbors(image, corners, window_size):
    # we use this function to get an array with all the neighborhoods
    # this is so we don't need to do a nested for loop in the matching function
    nh = []
    new_corners = []
    for corner in corners:
        if corner[1] - window_size//2 >= 0 and corner[1] + window_size//2 < image.shape[0] and corner[0] - window_size//2 >= 0 and corner[0] + window_size//2 < image.shape[1]:
            # need to check if it is a valid neighborhood, eg. all the pixel positions are within the borders of the image
            mat = image[corner[1]-window_size//2:corner[1]+window_size//2+1,corner[0]-window_size//2:corner[0]+window_size//2+1]
            new_corners.append(corner)
            nh.append(mat)
    return np.array(new_corners), np.array(nh)

def compute_ssd(template1,template2):
    # compute the ssd, the sum is specified in those axis so we can use broadcasting
    return ((template1-template2)**2).sum(axis=(1,2))

def compute_ncc(template1,template2):
    # computes 1-ncc, the mean and sum are specified in those axis to make use of broadcasting in the matching function
    return 1-(((template1 - template1.mean(axis=(1,2)))*(template2 - template2.mean(axis=(1,2))[:,None,None])).sum(axis=(1,2))/
            np.sqrt(((template1 - template1.mean(axis=(1,2)))**2).sum(axis=(1,2))*((template2 - template2.mean(axis=(1,2))[:,None,None])**2).sum(axis=(1,2)))    
            )

def matching(image_1, corners_1, image_2, corners_2, window_size=5, mode="ssd"):
    #first get the normalized grayscale version of each
    image_1 = normalize(grayscale(image_1))
    image_2 = normalize(grayscale(image_2))
    matches = []
    # get all the neighborhoods of the 2nd image with get_neighbors,
    # we do this to prevent the nested loop, so we turn it into a matrix multiplication instead with numpy broadcasting
    corners_2_, neighbors = get_neighbors(image_2, corners_2, window_size)
    # set the metric function
    metric = compute_ssd if mode=="ssd" else compute_ncc

    corresp_list = []
    # we now loop through the first image's detected corners, 
    # not sure how to vectorize this part
    for corner in corners_1:
        # get the neighborhood of the corner
        _, nh = get_neighbors(image_1,np.array([corner]),window_size)
        if nh.shape[0] > 0:
            mtr = metric(nh,neighbors)
            # we need the index where the minimum is
            corresp = np.argmin(mtr)
            if corresp not in corresp_list:
                # make sure that there is no other match that contains this corner
                corresp_list.append(corresp)
                matches.append([int(corner[0]),int(corner[1]),int(corners_2_[corresp][0]),int(corners_2_[corresp][1]),mtr[corresp]])
           
    matches = np.array(matches)
    # sort by descending, lower values the better the match
    matches_sorted = matches[matches[:,4].argsort()]
    return matches_sorted

def show_matches(image_1, image_2, matches, num_matches=100):
    # we are assuming the images have the same dimensions
    # this function just shows the first num_matches number of matches  
    _, offset_x, _ = image_1.shape
    # we combine images easily with hstack
    combined_img = np.hstack((image_1, image_2))
    # randomize the colors for the line that represents the match, we make it into this shape 
    # so that with enumerate we can just specify the index and use it
    colors = np.random.randint(0,256,(num_matches,3))
    for idx, match in enumerate(matches[:num_matches]):
        # use the randomized color by making that part of the array into a tuple, won't work otherwise
        color = (int(colors[idx,0]),int(colors[idx,1]),int(colors[idx,2]))
        combined_img = cv2.line(combined_img,(int(match[0]),int(match[1])),(int(match[2]+offset_x),int(match[3])),color,1)
    return combined_img     

def sift_features_match(image_1, image_2):
    # create sift feature detector
    sift = cv2.SIFT_create(400)

    # we create copies of both images to draw the keypoints on them
    image_1_draw = image_1.copy()
    image_2_draw = image_2.copy()
    # get the grayscale version of both
    img_1 = grayscale(image_1).astype(np.uint8)
    img_2 = grayscale(image_2).astype(np.uint8)
    # get both the keypoints and descriptors,
    # we use the descriptors for the matching part
    kp_1, desc_1 = sift.detectAndCompute(img_1,None)
    kp_2, desc_2 = sift.detectAndCompute(img_2,None)

    # we draw the keypoints with this function instead of cv2.circle since this function
    # can do fractional as opposed to cv2.circle which will give you an error
    image_1_draw = cv2.drawKeypoints(image_1_draw, kp_1, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    image_2_draw = cv2.drawKeypoints(image_2_draw, kp_2, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # I'm working with https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html for the brute force matcher
    bfmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bfmatcher.match(desc_1,desc_2)
    # sort in descending order
    matches = sorted(matches, key = lambda x:x.distance)
    # use this function to draw the matches since it can do fractional as well, we only show the top 100 matches
    img_3 = cv2.drawMatches(image_1,kp_1,image_2,kp_2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return image_1_draw, image_2_draw, img_3

image_list = [["temple_1","temple_2"],["hovde_2","hovde_3"],["keyboard_1","keyboard_2"],["building_1","building_2"]]
for pair in image_list:
    # this for loop does Harris corner detection with the list of scales specified below, 
    # does the NCC and SSD matching and draws it for each of the window sizes specified in num
    # and at the end it does the sift part
    img_name_1 = pair[0]
    img_name_2 = pair[1]
    image_1 = cv2.imread(img_name_1 + ".jpg")
    image_2 = cv2.imread(img_name_2 + ".jpg")
    # I tried with this scales in a grid search, but they can be edited to loop through any other scale
    scales = [0.8,1.2,1.6,2.0]
    # k=0.05 is the default in the function so this can either be erased or set with a different value
    k=0.05
    # this is the size of the window used for matching
    num = [41]
    # show the top 100 corners/matches
    show = 100
    for scale in scales:
        image_1_corners = harris_corner(image_1,scale,k=k)
        corn_img_1 = show_corners(image_1,image_1_corners,show)
        save_name_1 = f"IMAGES/{img_name_1}_sig{scale}_k{k}.jpg"
        cv2.imwrite(save_name_1,corn_img_1)

        image_2_corners = harris_corner(image_2,scale,k=k)
        corn_img_2 = show_corners(image_2,image_2_corners,show)
        save_name_2 = f"IMAGES/{img_name_2}_sig{scale}_k{k}.jpg"
        cv2.imwrite(save_name_2,corn_img_2)
        for n in num:
            ncc_match = matching(image_1, image_1_corners, image_2, image_2_corners, n, mode="ncc")
            ncc_match_img = show_matches(image_1, image_2, ncc_match, num_matches=show)
            save_name_ncc = f"IMAGES/{img_name_1}_{img_name_2}_sig{scale}_k{k}_ncc_match_nh{n}.jpg"
            cv2.imwrite(save_name_ncc,ncc_match_img)

            ssd_match = matching(image_1, image_1_corners, image_2, image_2_corners, n, mode="ssd")
            ssd_match_img = show_matches(image_1, image_2, ssd_match, num_matches=show)
            save_name_ssd= f"IMAGES/{img_name_1}_{img_name_2}_sig{scale}_k{k}_ssd_match_nh{n}.jpg"
            cv2.imwrite(save_name_ssd,ssd_match_img)
    # sift part
    image_1_kp, image_2_kp, match_img = sift_features_match(image_1, image_2)
    save_name_1 = f"IMAGES/{img_name_1}_sift_kp.jpg"
    cv2.imwrite(save_name_1,image_1_kp)
    save_name_2 = f"IMAGES/{img_name_2}_sift_kp.jpg"
    cv2.imwrite(save_name_2,image_2_kp)
    save_name_match = f"IMAGES/{img_name_1}_{img_name_2}_sift_match.jpg"
    cv2.imwrite(save_name_match,match_img)




