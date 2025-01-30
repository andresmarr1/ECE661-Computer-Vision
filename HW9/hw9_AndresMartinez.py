import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json
from numpy.random import default_rng
# we use these to plot the camera poses
from mpl_toolkits.mplot3d.art3d import *
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib
from matplotlib.patches import ConnectionPatch

def add_number(array,n):
    # this function takes in an array and adds any number
    # this is mainly for adding a 0, for z=0 in the real life plane coordinates
    ones_np = np.ones(array.shape[:-1])[...,None]
    ones_np *= n
    # need to expand the dimensions by 1 to be able to concatenate it
    return np.append(array,ones_np,axis=-1)

def get_normalization_matrix(points):
    # follows from the logic
    centroid = points[:,None,:].mean(axis=0)[0]
    shifted_points = points - centroid
    dist = np.linalg.norm(shifted_points,axis=1).mean()
    scale = np.sqrt(2)/dist
    T = np.array([[scale, 0, -scale*centroid[0]],
                  [0, scale, -scale*centroid[1]],
                  [0, 0, 1]])
    return T

def get_normalized_points(points):
    # this is just to get the normalization matrix and get the normalized points
    T = get_normalization_matrix(points)
    normed_points = apply_homography(points, T)
    return normed_points

def point2point_fmatrix(domain, range_):
    # follows from the logic
    # we changed this function to accept an arbitrary number of points,
    # this is because most of the systems we will solve have more than 4 data points
    num_points = len(domain)
    mat1 = np.empty((0,9),dtype=float)

    for i in range(num_points):
        x = domain[i,0]
        x_prime = range_[i,0]

        y = domain[i,1]
        y_prime = range_[i,1]

        mat1 = np.append(mat1,np.array([[x_prime*x, x_prime*y, x_prime, y_prime*x, y_prime*y, y_prime, x,y, 1]]),axis=0)
    return mat1
def linear_least_squares_homogeneous(mat1):
    # and this is for homogeneous least squares
    _, _, v_t = np.linalg.svd(mat1)
    return v_t[-1]

def get_F_matrix(domain,range_):
    # this is to obtain our F matrix
    T_left = get_normalization_matrix(domain)
    T_right = get_normalization_matrix(range_)
    mat1 = point2point_fmatrix(domain,range_)
    F = linear_least_squares_homogeneous(mat1)
    F = F.reshape((3,3))
    F = F/F[2,2]

    # condition the matrix
    u, d, v_t = np.linalg.svd(F)
    # zero the minimum eigenvalue, take advantage of the fact that svd gives them in decreasing order
    d[-1] = 0
    F_conditioned = u @ np.diag(d) @ v_t
    F_conditioned = T_right.T @ F_conditioned @ T_left
    F_conditioned /= F_conditioned[2,2]
    return F_conditioned
    
def get_epipoles(F):
    # this is to get the epipoles
    u, d, v_t = np.linalg.svd(F)
    e = v_t[-1]
    e_prime = u[:,-1]
    return e/e[2], e_prime/e_prime[2]

def skew_symmetric_matrix(array):
    # creates skew symmetric matrix, input has to be a (3,) shape vector
    return np.array([[0, -array[2],array[1]],
                     [array[2],0,-array[0]],
                     [-array[1],array[0],0]])

def get_canonical_P(F, e, e_prime):
    # follows from the logic
    P = np.eye(3)
    P = add_number(P,0)
    sk = skew_symmetric_matrix(e_prime)
    P_p = sk @ F
    #print(P_p, e_prime)
    # we only care about the first 3 columns of P
    P_prime = np.column_stack((P_p[:,0],P_p[:,1],P_p[:,2],e_prime))
    #print(P_prime)
    #print(np.linalg.matrix_rank(P),np.linalg.matrix_rank(P_prime))
    return P, P_prime

def triangulate(P, P_prime, domain, range_):
    # triangulation, follows from the logic, we also compute all the points using this
    num_points = len(domain)
    rl_coords = []
    P_1 = P[0,:]
    P_2 = P[1,:]
    P_3 = P[2,:]
    Pp_1 = P_prime[0,:]
    Pp_2 = P_prime[1,:]
    Pp_3 = P_prime[2,:]

    for i in range(num_points):
        x = domain[i,0]
        y = domain[i,1]
        x_prime = range_[i,0]
        y_prime = range_[i,1]
        A = np.array([(x*P_3) - P_1,
                      (y*P_3 )- P_2,
                      (x_prime*Pp_3) - Pp_1,
                      (y_prime*Pp_3) - Pp_2])
        X_rl = linear_least_squares_homogeneous(A)
        X_rl = X_rl/X_rl[3]
        rl_coords.append(X_rl)
    return np.array(rl_coords)[:,:3]

def reconstruct_matrix_worldpoints(params):
    # function used to reconstruct P and the triangulated world coordinates for our LM
    P_prime = params[:12].reshape((3,4))
    worldpoint_list = []
    for idx in range(12, len(params),3):
        X_x = params[idx]
        X_y = params[idx+1]
        X_z = params[idx+2]
        worldpoint = np.array([X_x,X_y,X_z])
        worldpoint_list.append(worldpoint)
    worldpoint_list = np.array(worldpoint_list)
    return P_prime, worldpoint_list

def get_params(P_prime, world_points):
    # get the parameters to be optimized using LM
    params = []
    P_p_flat = P_prime.flatten()
    for p in P_p_flat:
        params.append(p)
    for point in world_points:
        params.append(point[0])
        params.append(point[1])
        params.append(point[2])
    return np.array(params)

def apply_homography(positions, H):
    # reused from hw5
    # this gets the homography transformation for all the coordinates in the image that we get from the get_positions function
    # we do it in a way that exploits broadcasting, so we don't need to use for loops
    temp_pos = add_number(positions,1)
    new_pos = (H @ temp_pos.T).astype(float)
    new_pos /= new_pos[-1,:]
    new_pos = new_pos.T
    return new_pos[:,:-1]

def error_f(params, P, domain, range_):
    # function to be optimized with LM
    P_prime, world_points = reconstruct_matrix_worldpoints(params)
    num_points = len(domain)
    error_total = []
    for i in range(num_points):
        rl_img1 = apply_homography(np.array([world_points[i]]), P)[0]
        rl_img2 = apply_homography(np.array([world_points[i]]), P_prime)[0]
        err_1 = np.abs(rl_img1 - domain[i])
        error_total.append(err_1[0])
        error_total.append(err_1[1])
        error_total.append(0)
        err_2 = np.abs(rl_img2 - range_[i])
        error_total.append(err_2[0])
        error_total.append(err_2[1])
        error_total.append(0)

    err_total = np.array(error_total)
    return err_total.ravel()

def lm_optim_Pprime(P, P_prime, domain, range_):
    # our LM function
    world_points = triangulate(P, P_prime, domain, range_)
    params = get_params(P_prime, world_points)
    params_op = least_squares(error_f, params, method='lm', args=(P, domain, range_),verbose=False).x
    P_prime_op, worldPts = reconstruct_matrix_worldpoints(params_op)
    P_prime_op = P_prime_op/np.linalg.norm(P_prime_op[:,-1])
    return P_prime_op

def get_F_refined(P, P_prime_op):
    # this is to obtain our refined F using the P_prime we got from LM
    e_op = P_prime_op[:,-1]
    sk = skew_symmetric_matrix(e_op)
    F_op = sk @ P_prime_op @ np.linalg.pinv(P)
    return F_op[:,:3]/F_op[2,2]

def get_translation_matrix(h, w):
    # translation matrix mentioned in the logic
    T = np.array([[1, 0, -w/2],
                  [0, 1, -h/2],
                  [0, 0, 1]])
    return T
def get_rotation_matrix(e, h, w):
    # rotation matrix for our rectifying homography, follows from the logic
    theta = np.arctan(-(e[1] - h/2)/(e[0] - w/2))
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R, theta
def get_transformation_epipole2infinity(e, theta, h, w):
    # G matrix, from logic
    f = np.abs((e[0] - w/2) * np.cos(theta) - (e[1] - h/2) * np.sin(theta))

    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [-1/f, 0, 1]])
    return G

def get_rectifying_homography(image, e):
    # this is to get the right homography, follows from the logic
    h, w, _ = image.shape
    T = get_translation_matrix(h, w)
    R, theta = get_rotation_matrix(e, h, w)
    G = get_transformation_epipole2infinity(e, theta, h, w)

    H_c = G @ R @ T
    center = np.array([[w/2,h/2]])
    rectified_center = apply_homography(center,H_c)[0]
    T_rc = np.array([[1, 0, w/2],
                     [0, 1, h/2],
                     [0,0,1]])
    H = T_rc @ H_c
    H /= H[2,2]
 
    return H

def point2point_Hleft(domain, range_):
    # modified form the other function to solve our left homography system
    # we changed this function to accept an arbitrary number of points,
    # this is because most of the systems we will solve have more than 4 data points
    num_points = len(domain)
    mat1 = np.empty((0,3),dtype=float)
    mat2 = np.empty((0,1),dtype=float)

    for i in range(num_points):
        x = domain[i,0]
        x_prime = range_[i,0]
        y = domain[i,1]
        y_prime = range_[i,1]

        mat1 = np.append(mat1,np.array([[x, y, 1.]]),axis=0)
        mat2 = np.append(mat2, x_prime)
    
    return mat1, mat2[:,None]

def linear_least_squares(mat1, mat2):
    res = np.linalg.inv(mat1.T @ mat1) @ mat1.T @ mat2
    return res


    
def get_left_homography(P, P_prime, image_left, img_left_pts, img_right_pts, H_right):
    # this is to get our left homography
    h, w, _ = image_left.shape
    M = P_prime @ np.linalg.pinv(P)
    H_ini = H_right @ M
    H_ini /= H_ini[2,2]
    x = apply_homography(img_left_pts, H_ini)
    x_prime = apply_homography(img_right_pts, H_right)

    m1, m2 = point2point_Hleft(x, x_prime)
    H_abc_ = linear_least_squares(m1, m2)
    H_abc = np.eye(3)
    H_abc[0,:] = H_abc_[:,0]
    H_final = H_abc @ H_ini
    H_final /= H_final[2,2]
    
    return H_final

# the functions here are all for rectifying our images
def get_new_corners(img,H):
    # get the new corners after transformation
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    corners = np.array([[0,0],[w,0],[w,h],[0,h]])
    new_corners = apply_homography(corners,H)
    return new_corners
def paint_image(panorama,image, old_positions, new_positions,x_min,y_min):
    # reused from hw5, it is to paint the blank canvas
    #print(new_positions)
    for idx, n_pos in enumerate(new_positions):
        if 0 <= old_positions[idx,0] < image.shape[1] and 0 <= old_positions[idx,1] < image.shape[0]: #and 0 < new_positions[idx,1] - x_min < panorama.shape[1] and 0 <: 
            panorama[new_positions[idx,1] - y_min,new_positions[idx,0] - x_min] = image[int(old_positions[idx,1]),int(old_positions[idx,0])]
    return panorama
def get_binary_mask(image, H):
    # reused from hw5
    # we want these masks so we only iterate through the parts with the mask,
    # otherwise we need to iterate through the whole final panorama, which we don't want
    new_corners = get_new_corners(image,H)
    new_corners = np.ceil(new_corners).astype(int)
    x_min = new_corners[:,0].min()
    y_min = new_corners[:,1].min()
    new_corners[:,0] -= x_min
    new_corners[:,1] -= y_min

    x_max = new_corners[:,0].max()
    y_max = new_corners[:,1].max()


    canvas = np.zeros((y_max, x_max))
    empty = np.zeros((y_max, x_max,3)).astype(np.uint8)

    canvas = cv2.fillPoly(canvas,pts=[new_corners],color=255)
    return canvas, empty, x_min, y_min

def rectify_images(img_left, img_right, H_left, H_right):
    # recycled from hw5, somewhat
    # first get the blank canvas and also the roi map to project the image onto it
    # first we get all the corners, and get the new image shapes from these
    if len(img_left.shape) == 2:
        c = None
    else:
        c = 3
    corners_left = np.ceil(get_new_corners(img_left,H_left)).astype(int)
    x_min_left = corners_left[:,0].min()
    x_max_left = corners_left[:,0].max()
    w_left = x_max_left - x_min_left

    y_min_left = corners_left[:,1].min()
    y_max_left = corners_left[:,1].max()
    h_left = y_max_left - y_min_left

    corners_right = np.ceil(get_new_corners(img_right,H_right)).astype(int)

    x_min_right = corners_right[:,0].min()
    x_max_right = corners_right[:,0].max()
    w_right = x_max_right - x_min_right

    y_min_right = corners_right[:,1].min()
    y_max_right = corners_right[:,1].max()
    h_right = y_max_right - y_min_right

    x_min = np.array([x_min_left, x_min_right]).min()
    y_min = np.array([y_min_left, y_min_right]).min()

    x_max = np.array([x_max_left, x_max_right]).max()
    y_max = np.array([y_max_left, y_max_right]).max()

    w_final = x_max - x_min
    h_final = y_max - y_min
    if c == None:
        canvas_left = np.zeros((h_final, w_final)).astype(np.uint8)
    else:
        canvas_left = np.zeros((h_final, w_final, 3)).astype(np.uint8)
    roi_left = np.zeros((h_final, w_final))
    corners_left[:,0] -= x_min
    corners_left[:,1] -= y_min
    roi_left = cv2.fillPoly(roi_left,pts=[corners_left],color=255)
    
    if c == None:
        canvas_right = np.zeros((h_final, w_final)).astype(np.uint8)
    else:
        canvas_right = np.zeros((h_final, w_final, 3)).astype(np.uint8)
    roi_right = np.zeros((h_final, w_final))
    corners_right[:,0] -= x_min 
    corners_right[:,1] -= y_min
    roi_right = cv2.fillPoly(roi_right,pts=[corners_right],color=255)

    new_positions_left = (np.argwhere(roi_left == 255) + np.array([[y_min, x_min]]))[:,[1,0]]
    old_positions_left = apply_homography(new_positions_left, np.linalg.inv(H_left))

    new_positions_right = (np.argwhere(roi_right == 255) + np.array([[y_min, x_min]]))[:,[1,0]]
    old_positions_right = apply_homography(new_positions_right, np.linalg.inv(H_right))


    rect_left = paint_image(canvas_left, img_left, old_positions_left, new_positions_left, x_min, y_min)
    rect_right = paint_image(canvas_right, img_right, old_positions_right, new_positions_right, x_min, y_min)
    # we also return the offset as we will make use of it later
    return rect_left, rect_right, x_min, y_min

def plot_correspondences(image_l, image_r, l_pts, r_pts, figname):
    # this is a modified function from superglue_ece661.py, plot_keypoints in particular
    # reused from hw5
    tot_img = np.hstack((image_l, image_r))
    tot_img = cv2.cvtColor(tot_img, cv2.COLOR_BGR2RGB)
    _, w0, _ = image_l.shape
    ms = 0.5
    lw = 0.5

    plt.figure()
    #plt.clf()
    plt.imshow(tot_img);
    # plot inliers in green (match and keypoint)
    for ikp0 in range(len(l_pts)):
        color_random = np.random.random(3)
        plt.plot(l_pts[ikp0,0], l_pts[ikp0,1], marker="o",color=color_random, markersize=ms)
        plt.plot(r_pts[ikp0,0]+w0, r_pts[ikp0,1], marker="o", color=color_random, markersize=ms)         
        plt.plot((l_pts[ikp0,0], r_pts[ikp0,0]+w0), (l_pts[ikp0,1], r_pts[ikp0,1]), color=color_random,linestyle='--', linewidth=lw, markersize=ms)
    plt.axis('off');
    plt.savefig(figname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return

# since we used labelme, the points are saved as json dict, so we use this to load them
with open('image_left.json', "r") as f:
    data = json.load(f)

img_left_points = np.array(data["shapes"][0]["points"])

with open('image_right.json', "r") as f:
    data = json.load(f)
    
img_right_points = np.array(data["shapes"][0]["points"])

img_left = cv2.imread("image_left.jpeg")
img_right = cv2.imread("image_right.jpeg")

# get correspondences
plot_correspondences(img_left, img_right, img_left_points, img_right_points, '8point_correspondence.png')
# get F and normalize it
F_m = get_F_matrix(img_left_points, img_right_points)
e, e_prime = get_epipoles(F_m)
P, P_prime = get_canonical_P(F_m, e, e_prime)
P_prime_op = lm_optim_Pprime(P, P_prime, img_left_points, img_right_points)
F_op = get_F_refined(P, P_prime_op)

F_op /= F_op[2,2]
# get the epipoles
e_op, e_prime_op = get_epipoles(F_op)
# now the left and right homographies
H_right = get_rectifying_homography(img_right, e_prime_op)
H_right /= H_right[2,2]

H_left = get_left_homography(P, P_prime_op, img_left, img_left_points, img_right_points, H_right)
H_left /= H_left[2,2]

rect_left, rect_right, x_offset, y_offset = rectify_images(img_left, img_right, H_left, H_right)

together = np.hstack((rect_left, rect_right))
cv2.imwrite("both_rectified.png",together)

rect_pts_left = apply_homography(img_left_points,H_left)

rect_pts_right = apply_homography(img_right_points,H_right)
# this is where we use the offsets
rect_pts_left -= np.array([[x_offset, y_offset]])
rect_pts_right -= np.array([[x_offset, y_offset]])

plot_correspondences(rect_left, rect_right, rect_pts_left, rect_pts_right, '8point_correspondence_rectified.png')

gray_left = cv2.imread("image_left.jpeg", cv2.IMREAD_GRAYSCALE)
gray_right = cv2.imread("image_right.jpeg", cv2.IMREAD_GRAYSCALE)
canny_p1=50
canny_p2=150
edges_l = cv2.Canny(gray_left,canny_p1,canny_p2)
edges_r = cv2.Canny(gray_right,canny_p1,canny_p2)
# rectify the canne output as explained in the report
rect_ed_l, rect_ed_r, x_offset, y_offset = rectify_images(edges_l, edges_r, H_left, H_right)
together_canny = np.hstack((rect_ed_l, rect_ed_r))
cv2.imwrite("rectified_cannys.png",together_canny)
# the next few functions are for the correspondences
def get_neighbours(image_r, image_gray_r, pos, window_size=9, delta=2):
    # reused from hw4
    # this might not be the best way to do it, but this is the fastest idea I could come up with
    # first we get the nonzero positions in image_r
    image_r_nonzero = np.argwhere(image_r != 0)
    # now we check which ones are within a similar row to the position we are looking for
    image_r_filtered = np.argwhere(np.abs(image_r_nonzero[:,0] - pos[0]) < delta)
    image_r_f = image_r_nonzero[image_r_filtered][:,0,:]
    nh = []
    pos_list = []
    for position in image_r_f:
        idx = position[1]
        jdx = position[0]
        window = image_gray_r[jdx-window_size//2:jdx+window_size//2+1,idx-window_size//2:idx+window_size//2+1]
        nh.append(window)
        pos_list.append(position)
    return np.array(nh), np.array(pos_list)
def compute_ssd(template,templates):
    # this is to get the SSD for all 3 channels, and then average them at the end
    #template = template[None,:]
    
    # Compute the squared differences between the single template and each template
    ssd = ((templates - template)**2 )
    ssd_ = ssd.sum(axis=(1,2,3)) 

    return ssd_

image_l_nonzero = np.argwhere(rect_ed_l != 0)

gray_rect_left = cv2.cvtColor(rect_left,cv2.COLOR_BGR2GRAY)
gray_rect_right = cv2.cvtColor(rect_right,cv2.COLOR_BGR2GRAY)

window_size = 25
x_tol = 30
delta=3
# perform ssd matching, delta is just the number of extra rows we scan, 2 up 2 down
matches = []
for l_pos in image_l_nonzero:
    neighs, neigh_pos_list = get_neighbours(rect_ed_r, rect_right,l_pos, window_size=window_size, delta=delta)
    #print(neighs.shape)
    window_l = rect_left[l_pos[0] - window_size//2:l_pos[0] + window_size//2 + 1, l_pos[1] - window_size//2:l_pos[1] - window_size//2 + 1][None,...]
    
    try:
        ssd_match = compute_ssd(window_l, neighs)
        mat = np.argmin(ssd_match)
        if np.abs(l_pos[1] - neigh_pos_list[mat,1]) < x_tol:
        
            matches.append([l_pos[0],l_pos[1], neigh_pos_list[mat,0],neigh_pos_list[mat,1], ssd_match[mat]])
        else:
            
            continue
    except:
        continue


matches = np.array(matches)
sorted_matches = matches[matches[:, 4].argsort()]
# change the y and x due to image coordinates
l_match = sorted_matches[:,[1,0]]
r_match = sorted_matches[:,[3,2]]



rng = default_rng()
numbers = rng.choice(3800, size=500, replace=False) # pick 500 at random

plot_correspondences(rect_left, rect_right, l_match[numbers], r_match[numbers], 'test_matches_rectified.png')

# get x and y from the original image
# first consider the offset

l_matches = l_match[numbers]
l_matches[:,0] += x_offset 
l_matches[:,1] += y_offset
r_matches = r_match[numbers]
r_matches[:,0] += x_offset
r_matches[:,1] += y_offset

# we can project these into our original image using the inverse homographies

l_matches_og = apply_homography(l_matches, np.linalg.inv(H_left))
r_matches_og = apply_homography(r_matches, np.linalg.inv(H_right))

plot_correspondences(img_left, img_right, l_matches_og, r_matches_og, 'test_matches_projectedog.png')
# the following functions are to optimize only the world coordinates
def triangulate(P, P_prime, domain, range_):

    num_points = len(domain)
    rl_coords = []
    P_1 = P[0,:]
    P_2 = P[1,:]
    P_3 = P[2,:]
    Pp_1 = P_prime[0,:]
    Pp_2 = P_prime[1,:]
    Pp_3 = P_prime[2,:]

    for i in range(num_points):
        x = domain[i,0]
        y = domain[i,1]
        x_prime = range_[i,0]
        y_prime = range_[i,1]
        A = np.array([(x*P_3) - P_1,
                      (y*P_3 )- P_2,
                      (x_prime*Pp_3) - Pp_1,
                      (y_prime*Pp_3) - Pp_2])
        X_rl = linear_least_squares_homogeneous(A)
        X_rl = X_rl/X_rl[-1]
        rl_coords.append(X_rl[:3])
    return np.array(rl_coords)[:,:3]

def reconstruct_worldpoints(params):
    worldpoint_list = []
    for idx in range(0,len(params),3):
        X_x = params[idx]
        X_y = params[idx+1]
        X_z = params[idx+2]
        worldpoint = np.array([X_x,X_y,X_z])
        worldpoint_list.append(worldpoint)
    worldpoint_list = np.array(worldpoint_list)
    return worldpoint_list

def get_params_wp(world_points):
    params = []
    for point in world_points:
        params.append(point[0])
        params.append(point[1])
        params.append(point[2])
    return np.array(params)

def error_f_wp(params, P, P_prime, domain, range_):
    world_points = reconstruct_worldpoints(params)
    num_points = len(domain)
    error_total = []
    for i in range(num_points):
        rl_img1 = apply_homography(np.array([world_points[i]]), P)[0]
        rl_img2 = apply_homography(np.array([world_points[i]]), P_prime)[0]
        err_1 = np.abs(rl_img1 - domain[i])
        error_total.append(err_1[0])
        error_total.append(err_1[1])
        err_2 = np.abs(rl_img2 - range_[i])
        error_total.append(err_2[0])
        error_total.append(err_2[1])

    err_total = np.array(error_total)
    return err_total.ravel()

def lm_optim_wp(P, P_prime, domain, range_):

    world_points = triangulate(P, P_prime, domain, range_)
    
    params = get_params_wp(world_points)
    params_op = least_squares(error_f_wp, params, method='lm', args=(P,P_prime, domain, range_),verbose=True).x
    worldPts = reconstruct_worldpoints(params_op)
    return worldPts
# we also place the original correspondences
total_left_pts = np.vstack((l_matches_og, img_left_points))

total_right_pts = np.vstack((r_matches_og, img_right_points))

world_pts = lm_optim_wp(P, P_prime_op, total_left_pts, total_right_pts)

norm_max = world_pts.max()

world_pts = world_pts/norm_max
# these are the original world points
world_pts_og = world_pts[-8:]

def get_camera_pose(P,norm_max):
    # follows from hw8
    K = np.eye(3)
    R_t = np.linalg.inv(K) @ P
    R = R_t[:,:3]
    # we noticed that R was not orthonormal, so we condition it to be orthonormal here
    u, _, v = np.linalg.svd(R)
    R_conditioned = u @ v
    t = R_t[:,3]
    #R = R/R[2,2]
    C = - R_conditioned.T @ t

    X_xcam = np.array([1,0,0])
    X_ycam = np.array([0,1,0])
    X_zcam = np.array([0,0,1])

    X_x = R_conditioned.T @ X_xcam 
    X_y = R_conditioned.T @ X_ycam
    X_z = R_conditioned.T @ X_zcam


    return C/norm_max, X_x, X_y, X_z

c, dx_x, dx_y, dx_z = get_camera_pose(P,norm_max)
c_prime, dx_x_prime, dx_y_prime, dx_z_prime = get_camera_pose(P_prime_op,norm_max)



fig = plt.figure(figsize=(11,11))
ax1 = fig.add_subplot(111, projection='3d')
ms = 5
lw = 0.5
color_random = np.random.random((8,3))
for idx in range(len(world_pts)-8):
    ax1.scatter(world_pts[idx,0],world_pts[idx,1],world_pts[idx,2], marker='o', color='g', s=2)
for idx in range(len(world_pts_og)):
    ax1.scatter(world_pts_og[idx,0],world_pts_og[idx,1],world_pts_og[idx,2], marker='o', color=color_random[idx], s=7)
#camera 1
color_random_ = np.random.random(3)
center = c
c_x, c_y, c_z = center
x_dir = dx_x/10
x_x, x_y, x_z = x_dir
y_dir = dx_y/10
y_x, y_y, y_z = y_dir
z_dir = dx_z/10
z_x, z_y, z_z = z_dir

# this is for our direction vectors
ax1.quiver(c_x, c_y, c_z, x_x, x_y, x_z, color='r', linewidth=1)
ax1.quiver(c_x, c_y, c_z, y_x, y_y, y_z, color='g', linewidth=1)
ax1.quiver(c_x, c_y, c_z, z_x, z_y, z_z, color='b', linewidth=1)

# now the camera plane, in here we notice that the corners are made up from addition and substractions of the camera center and the directions,
# so we make use to that
points = [center + x_dir + y_dir,
        center + x_dir - y_dir,
        center - x_dir - y_dir,
        center - x_dir + y_dir]
    
poly = Poly3DCollection([points], color='cyan', alpha=0.4, edgecolor=None)
ax1.add_collection3d(poly)

#cam 2
color_random_ = np.random.random(3)
center = c_prime
c_x, c_y, c_z = center
x_dir = dx_x_prime/10
x_x, x_y, x_z = x_dir
y_dir = dx_y_prime/10
y_x, y_y, y_z = y_dir
z_dir = dx_z_prime/10
z_x, z_y, z_z = z_dir
# this is for our direction vectors
ax1.quiver(c_x, c_y, c_z, x_x, x_y, x_z, color='r', linewidth=1)
ax1.quiver(c_x, c_y, c_z, y_x, y_y, y_z, color='g', linewidth=1)
ax1.quiver(c_x, c_y, c_z, z_x, z_y, z_z, color='b', linewidth=1)

# now the camera plane, in here we notice that the corners are made up from addition and substractions of the camera center and the directions,
# so we make use to that
points = [center + x_dir + y_dir,
        center + x_dir - y_dir,
        center - x_dir - y_dir,
        center - x_dir + y_dir]
    
poly = Poly3DCollection([points], color='purple', alpha=0.4, edgecolor=None)
ax1.add_collection3d(poly)

#ax1.view_init(90,-90)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)

plt.savefig(f'reconstruction_cameras.png', bbox_inches='tight', pad_inches=0)
plt.close()


fig = plt.figure(figsize=(11,11))
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312, projection='3d')
ax2 = fig.add_subplot(313)
ms = 5
lw = 0.5
ax0.imshow(cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB))
color_random = np.random.random((8,3))
for idx in range(len(world_pts)-8):
    ax1.scatter(world_pts[idx,0],world_pts[idx,1],world_pts[idx,2], marker='o', color='g', s=2)
for idx in range(len(world_pts_og)):
    ax0.plot(img_left_points[idx,0], img_left_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    ax1.scatter(world_pts_og[idx,0],world_pts_og[idx,1],world_pts_og[idx,2], marker='o', color=color_random[idx], s=7)
    ax2.plot(img_right_points[idx,0], img_right_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    

#ax1.view_init(0,0)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(-0.3,0.7)
ax1.set_ylim(-0.2,1.1)
ax1.set_zlim(-0.0005,0.0018)

for idx in range(len(world_pts_og)):
    x1, y1, _ = proj3d.proj_transform(world_pts_og[idx,0], world_pts_og[idx,1], world_pts_og[idx,2], ax1.get_proj())
    con = ConnectionPatch(
        xyA=(img_left_points[idx,0], img_left_points[idx,1]),  # Point in the first plot
        xyB=(x1, y1),  # Corresponding point in the second plot
        coordsA="data",      # Interpret xyA in data coordinates of ax1
        coordsB="data",      # Interpret xyB in data coordinates of ax2
        axesA=ax0,           # Reference the first axes
        axesB=ax1,           # Reference the second axes
        color=color_random[idx],        # Line color
        linestyle="--"       # Line style
    )
    # Add the connection to the second plot
    fig.add_artist(con)
    con1 = ConnectionPatch(
        xyA=(img_right_points[idx,0], img_right_points[idx,1]),  # Point in the first plot
        xyB=(x1, y1),  # Corresponding point in the second plot
        coordsA="data",      # Interpret xyA in data coordinates of ax1
        coordsB="data",      # Interpret xyB in data coordinates of ax2
        axesA=ax2,           # Reference the first axes
        axesB=ax1,           # Reference the second axes
        color=color_random[idx],        # Line color
        linestyle="--"       # Line style
    )
    fig.add_artist(con1)
    # Add the connection to the second plot

plt.savefig(f'reconstruction_view1.png', bbox_inches='tight', pad_inches=0)
plt.close()

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib
from matplotlib.patches import ConnectionPatch

fig = plt.figure(figsize=(11,11))
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312, projection='3d')
ax2 = fig.add_subplot(313)
ms = 5
lw = 0.5
ax0.imshow(cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB))
color_random = np.random.random((8,3))
for idx in range(len(world_pts)-8):
    ax1.scatter(world_pts[idx,0],world_pts[idx,1],world_pts[idx,2], marker='o', color='g', s=2)
for idx in range(len(world_pts_og)):
    ax0.plot(img_left_points[idx,0], img_left_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    ax1.scatter(world_pts_og[idx,0],world_pts_og[idx,1],world_pts_og[idx,2], marker='o', color=color_random[idx], s=7)
    ax2.plot(img_right_points[idx,0], img_right_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    

ax1.view_init(0,0)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(-0.3,0.7)
ax1.set_ylim(-0.2,1.1)
ax1.set_zlim(-0.0005,0.0018)

for idx in range(len(world_pts_og)):
    x1, y1, _ = proj3d.proj_transform(world_pts_og[idx,0], world_pts_og[idx,1], world_pts_og[idx,2], ax1.get_proj())
    #ax0.plot([img_left_points[idx,0], x1], [img_left_points[idx,1], y1], color=color_random, linestyle='--')
    con = ConnectionPatch(
        xyA=(img_left_points[idx,0], img_left_points[idx,1]),  # Point in the first plot
        xyB=(x1, y1),  # Corresponding point in the second plot
        coordsA="data",      # Interpret xyA in data coordinates of ax1
        coordsB="data",      # Interpret xyB in data coordinates of ax2
        axesA=ax0,           # Reference the first axes
        axesB=ax1,           # Reference the second axes
        color=color_random[idx],        # Line color
        linestyle="--"       # Line style
    )
    # Add the connection to the second plot
    fig.add_artist(con)
    con1 = ConnectionPatch(
        xyA=(img_right_points[idx,0], img_right_points[idx,1]),  # Point in the first plot
        xyB=(x1, y1),  # Corresponding point in the second plot
        coordsA="data",      # Interpret xyA in data coordinates of ax1
        coordsB="data",      # Interpret xyB in data coordinates of ax2
        axesA=ax2,           # Reference the first axes
        axesB=ax1,           # Reference the second axes
        color=color_random[idx],        # Line color
        linestyle="--"       # Line style
    )
    fig.add_artist(con1)
    # Add the connection to the second plot

plt.savefig(f'reconstruction_view2.png', bbox_inches='tight', pad_inches=0)
plt.close()

fig = plt.figure(figsize=(11,11))
ax0 = fig.add_subplot(311)
ax1 = fig.add_subplot(312, projection='3d')
ax2 = fig.add_subplot(313)
ms = 5
lw = 0.5
ax0.imshow(cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB))
color_random = np.random.random((8,3))
for idx in range(len(world_pts)-8):
    ax1.scatter(world_pts[idx,0],world_pts[idx,1],world_pts[idx,2], marker='o', color='g', s=2)
for idx in range(len(world_pts_og)):
    ax0.plot(img_left_points[idx,0], img_left_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    ax1.scatter(world_pts_og[idx,0],world_pts_og[idx,1],world_pts_og[idx,2], marker='o', color=color_random[idx], s=7)
    ax2.plot(img_right_points[idx,0], img_right_points[idx,1], marker="o",color=color_random[idx], markersize=ms)
    x1, y1, _ = proj3d.proj_transform(world_pts_og[idx,0], world_pts_og[idx,1], world_pts_og[idx,2], ax1.get_proj())
    print(x1,y1)
    

ax1.view_init(60,45)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.set_xlim(-0.001,0.01)
ax1.set_ylim(0.002,0.010)
ax1.set_zlim(0,0.00002)

for idx in range(len(world_pts_og)):
    x1, y1, _ = proj3d.proj_transform(world_pts_og[idx,0], world_pts_og[idx,1], world_pts_og[idx,2], ax1.get_proj())
    if x1 < 0.0711:
        #ax0.plot([img_left_points[idx,0], x1], [img_left_points[idx,1], y1], color=color_random, linestyle='--')
        con = ConnectionPatch(
            xyA=(img_left_points[idx,0], img_left_points[idx,1]),  # Point in the first plot
            xyB=(x1, y1),  # Corresponding point in the second plot
            coordsA="data",      # Interpret xyA in data coordinates of ax1
            coordsB="data",      # Interpret xyB in data coordinates of ax2
            axesA=ax0,           # Reference the first axes
            axesB=ax1,           # Reference the second axes
            color=color_random[idx],        # Line color
            linestyle="--"       # Line style
        )
        # Add the connection to the second plot
        fig.add_artist(con)
        con1 = ConnectionPatch(
            xyA=(img_right_points[idx,0], img_right_points[idx,1]),  # Point in the first plot
            xyB=(x1, y1),  # Corresponding point in the second plot
            coordsA="data",      # Interpret xyA in data coordinates of ax1
            coordsB="data",      # Interpret xyB in data coordinates of ax2
            axesA=ax2,           # Reference the first axes
            axesB=ax1,           # Reference the second axes
            color=color_random[idx],        # Line color
            linestyle="--"       # Line style
        )
        fig.add_artist(con1)
    # Add the connection to the second plot

plt.savefig(f'reconstruction_view3.png', bbox_inches='tight', pad_inches=0)
plt.close()

##### THIS IS FOR TASK 3
def census_transformation(img_left, img_right, window_size, d_Max, left=True):
    # we want to pad both images, to not lose out on those borders pixels, however since we are moving from 0 to d_Max we need to consider d_Max in our padding
    # so then the logical choice is to pad by d_Max + half the window size
    h,w = img_left.shape 
    disp_map = np.zeros_like(img_left)
    border_size = d_Max + window_size//2
    # technically, if want to compute the left disparity match, we only need to pad the right side of the left image, and the left side of the right image, as we go from the position to d_Max in the right image
    # but since this implementation can be used for both the left or right image, we pad left and right sides.
    # As for the top and bottom padding, we only need to make these half the window size, since the don't "move" in that direction
    img_right = cv2.copyMakeBorder(img_right, window_size//2, window_size//2, border_size, border_size, cv2.BORDER_CONSTANT, 0)
    img_left = cv2.copyMakeBorder(img_left, window_size//2, window_size//2, border_size, border_size, cv2.BORDER_CONSTANT, 0)
    # the way to compute the disparity is a bit different from left and right, so we make the choice here
    if left:
        # like I mentioned, the rows will go from window_size//2 to img_left.shape[0] - window_size//2, to account for the padding
        for jdx in range(window_size//2, h+window_size//2):
            # for the x position, we need to consider the border size we had mentioned, 
            for idx in range(border_size, w+border_size):
                # save each error in this array
                # get the left window
                left_window = img_left[(jdx - window_size//2):(jdx + window_size//2 + 1), (idx - window_size//2):(idx + window_size//2 + 1)]
                center_L = img_left[jdx,idx]
                left_window = left_window.ravel()
                # this is our descriptor bitvector
                bit_L = left_window > center_L
                err = []
                for d in range(d_Max+1):
                    # this is where we go from 0 to d_Max in the right image, notice that only the x direction is changing
                    right_window = img_right[(jdx - window_size//2):(jdx + window_size//2 + 1),(idx - d - window_size//2):(idx - d + window_size//2 +1)]
                    right_window = right_window.ravel()
                    center_R = img_right[jdx,idx - d]
                    bit_R = right_window > center_R
                    # logical xor

                    diff = np.logical_xor(bit_L, bit_R)
                    s_diff = diff.sum()
                    err.append(s_diff)
                # and now all that is left is to take the minimum sum value
                disp_map[jdx-window_size//2,idx-border_size] = np.argmin(np.array(err))
    else:
        # we do a similar thing for the right image, however the difference is that in this case, we move in the opposite direction for 0 to d_Max
        for jdx in range(window_size//2, h+window_size//2):
            for idx in range(border_size, w+border_size):
                err = []
                right_window = img_right[(jdx - window_size//2):(jdx + window_size//2 + 1), (idx - window_size//2):(idx + window_size//2 + 1)]
                center_R = img_right[jdx,idx]
                right_window = right_window.ravel()
                bit_R = right_window > center_R
                err = []
                for d in range(d_Max+1):
                    left_window = img_left[(jdx - window_size//2):(jdx + window_size//2 + 1),(idx + d - window_size//2):(idx + d + window_size//2 +1)]
                    left_window = left_window.ravel()
                    center_L = img_left[jdx,idx + d]
                    bit_L = left_window > center_L

                    diff = np.logical_xor(bit_R, bit_L)
                    s_diff = diff.sum()
                    err.append(s_diff)
                disp_map[jdx-window_size//2,idx-border_size] = np.argmin(np.array(err))
    return disp_map.astype(np.uint8)

def check_error_disp(disparity_map, gt_map, delta=2):
    # get valid positions

    # Now the mask
    disp_mask = np.zeros_like(gt_map).astype(np.uint8)
    diffe = np.abs(disparity_map - gt_map)
    disp_mask[diffe < delta] = 255
    disp_mask[gt_map == 0] = 0

    #acc = (disp_mask == 255).sum()
    #acc = acc/(gt_map.shape[0]*gt_map.shape[1])
    acc = np.count_nonzero(disp_mask)/np.count_nonzero(gt_map)

    return acc, disp_mask
    

image_left = cv2.imread("Task3Images/im2.png", cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread("Task3Images/im6.png", cv2.IMREAD_GRAYSCALE)

gt_left = cv2.imread("Task3Images/disp2.png", cv2.IMREAD_GRAYSCALE)
gt_right = cv2.imread("Task3Images/disp6.png", cv2.IMREAD_GRAYSCALE)

gt_left = gt_left.astype(np.float32)/4
gt_left = gt_left.astype(np.uint8)

gt_right = gt_right.astype(np.float32)/4
gt_right = gt_right.astype(np.uint8)

d_max_left = gt_left.max()
d_max_right = gt_right.max()


window_sizes = [5,15,25,35]
for w in window_sizes:
    print("-"*10)
    print(f"Window size is {w}")
    dispmap_left = census_transformation(image_left, image_right, w, d_max_left, left=True)
    dispmap_right = census_transformation(image_left, image_right, w, d_max_right, left=False)
    acc_left, disp_mask_left = check_error_disp(dispmap_left, gt_left, delta=2)
    acc_right, disp_mask_right = check_error_disp(dispmap_right, gt_right, delta=2)

    print(f"Left accuracy is {acc_left}")
    print(f"Right accuracy is {acc_right}")

    plt.imshow(dispmap_left,cmap="gray")
    plt.axis("off")
    plt.savefig(f'images_report/disparity_map_left_wsize{w}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(dispmap_right,cmap="gray")
    plt.axis("off")
    plt.savefig(f'images_report/disparity_map_right_wsize{w}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(disp_mask_left,cmap="gray")
    plt.axis("off")
    plt.savefig(f'images_report/left_diff_wsize{w}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(disp_mask_right,cmap="gray")
    plt.axis("off")
    plt.savefig(f'images_report/right_diff_wsize{w}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
