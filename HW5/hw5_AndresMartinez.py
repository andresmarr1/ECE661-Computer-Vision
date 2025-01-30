# we use superpoint so this needs to be inside the supergluepretrainednetwork folder
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from superglue_ece661 import *
from models.matching import Matching
from models.utils import process_resize, read_image
from scipy.optimize import least_squares

def point_to_point_system(domain, range_, num_points):
    # we changed this function to accept an arbitrary number of points,
    # this is because most of the systems we will solve have more than 4 data points
    mat1 = np.empty((0,8),dtype=float)
    mat2 = np.empty((0,0),dtype=float)
    for i in range(num_points):
        x = domain[i,0]
        x_prime = range_[i,0]
        y = domain[i,1]
        y_prime = range_[i,1]

        mat1 = np.append(mat1,np.array([[x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime]]),axis=0)
        mat1 = np.append(mat1,np.array([[0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]]),axis=0)
        mat2 = np.append(mat2,x_prime)
        mat2 = np.append(mat2,y_prime)

    return mat1,mat2.reshape(num_points*2,1)

def get_H_matrix(domain,range_,n_points):
    # we no longer use this, since all the systems we will solve in this homework are
    # overdetermined, in both ransac and least squares
    mat1, mat2 = point_to_point_system(domain, range_,n_points)
    # we use this function to solve the equation described in the logic, 
    # I replaced np.dot with @ as that is what they suggest in the numpy website
    sol = np.linalg.inv(mat1) @ mat2
    # we append the 1 since this will only have 8 values, the 1 is missing
    sol = np.append(sol,np.array([[1]]),axis=0)
    return sol.reshape((3,3)).astype(float)

def get_H_matrix_overdetermined(domain,range_,n_points):
    # this is the one we use, in both ransac and least squares, it takes in n_points datapoints and calculates the matrix
    mat1, mat2 = point_to_point_system(domain, range_,n_points)
    # we use this function to solve the equation described in the logic, 
    # I replaced np.dot with @ as that is what they suggest in the numpy website
    # in this case we use the pseudo inverse (ATA)^-1 AT
    sol = np.linalg.inv(mat1.T @ mat1) @ mat1.T @ mat2
    # we append the 1 since this will only have 8 values, the 1 is missing
    sol = np.append(sol,np.array([[1]]),axis=0)
    return sol.reshape((3,3)).astype(float)

def add_ones(array):
    # this function takes in an array and adds ones
    # this is mainly for applying homographies
    ones_np = np.ones(array.shape[:-1])[...,None]
    # need to expand the dimensions by 1 to be able to concatenate it
    return np.append(array,ones_np,axis=-1)

def apply_homography(positions, H):
    # this gets the homography transformation for all the coordinates in the image that we get from the get_positions function
    # we do it in a way that exploits broadcasting, so we don't need to use for loops
    temp_pos = add_ones(positions)
    new_pos = (H @ temp_pos.T).astype(float)
    new_pos /= new_pos[2,:]
    return new_pos[:2,:].T

def check_inliers(kp, kp_compare, tr_kp, delta):
    # we compare using the l2 norm, the distance between the transformed keypoints (tr_kp) and the true keypoints (kp_compare)
    dists = np.linalg.norm(tr_kp - kp_compare, axis=1)
    inliers_1 = []
    inliers_2 = []
    outliers_1 = []
    outliers_2 = []
    # loop through it and check if the distance is less than delta,
    # append both image 1 and image 2 keypoints to inliers if it is, 
    # append to outliers if it isn't
    inliers_idx = np.argwhere(dists < delta)
    outliers_idx = np.argwhere(dists >= delta)
    # again, we don't need to use for loops here
    inliers_1 = np.take(kp, inliers_idx,axis=0)[:,0,:]
    inliers_2 = np.take(kp_compare, inliers_idx,axis=0)[:,0,:]
    outliers_1 = np.take(kp, outliers_idx,axis=0)[:,0,:]
    outliers_2 = np.take(kp_compare, outliers_idx,axis=0)[:,0,:]
    return inliers_1, inliers_2, outliers_1, outliers_2

def ransac(kp_1, kp_2 , sigma=2, epsilon=0.1, n_points=20):
    # we run ransac here, we take into consideration the scale (sigma), epsilon and the number of points
    # from these values we get the other parameters like delta, N and M
    # our epsilon is 0.1 since superpoint doesn't seem to generate many outliers, so we set it to a low value
    num_keypoints = len(kp_1)
    p = 0.99
    delta = sigma * 3
    N = int(np.log(1-p)/np.log(1- (1 - epsilon)**n_points))
    M = int((1 - epsilon)*num_keypoints)
    random_choose = np.random.randint(0,num_keypoints,(N,n_points))
    
    for rand in random_choose:
        domain = np.take(kp_1,rand,axis=0)
        range_ = np.take(kp_2,rand,axis=0)
        H = get_H_matrix_overdetermined(domain,range_,n_points)

        h_kp_1 = apply_homography(kp_1,H)

        inliers_1, inliers_2, outliers_1, outliers_2 = check_inliers(kp_1, kp_2, h_kp_1, delta)
        if len(inliers_1) >= M:
            return np.array(inliers_1), np.array(inliers_2), np.array(outliers_1), np.array(outliers_2)
        
    return None, None, None, None

def get_new_corners(img,H):
    # we just get the new corners, all mapped to image 3
    # for image 1 and 2 there should be negatives, but that is exactly what we want
    # so we can get the actual size of the new image
    h, w, _ = img.shape
    corners = np.array([[0,0],[w,0],[w,h],[0,h]])
    new_corners = apply_homography(corners,H)
    return new_corners

def paint_image(panorama,image, old_positions, new_positions,x_min,y_min):
    # this is to paint an image into the panorama, knowing the coordinates of where the binary mask is 255 (new_positions)
    # yes, its old positions even though technically, they are calculated after the others, it is just because its the coordinates from the
    # old image
    #print(new_positions)
    for idx, n_pos in enumerate(new_positions):
        if 0 <= old_positions[idx,0] < image.shape[1] and 0 <= old_positions[idx,1] < image.shape[0]: #and 0 < new_positions[idx,1] - x_min < panorama.shape[1] and 0 <: 
            panorama[new_positions[idx,1] - y_min,new_positions[idx,0] - x_min] = image[int(old_positions[idx,1]),int(old_positions[idx,0])]
    return panorama

def get_binary_mask(panorama, corners, x_min, y_min):
    # we want these masks so we only iterate through the parts with the mask,
    # otherwise we need to iterate through the whole final panorama, which we don't want
    canvas = np.zeros((panorama.shape[0],panorama.shape[1])).astype(np.uint8)
    new_corners = corners
    new_corners[:,0] -= x_min
    new_corners[:,1] -= y_min
    new_corners = new_corners.astype(np.int32)
    canvas = cv2.fillPoly(canvas,pts=[new_corners],color=255)
    return canvas

def error_f(H, kp0,kp1):
    # get the error X - f 
    kp_k = apply_homography(kp0, H.reshape(3,3))
    # we use ravel since we want it to be a flat array
    return (kp1 - kp_k).ravel()

def jacobian(H, kp0):
    # our jacobian matrix
    J = np.zeros((2*len(kp0),9))
    for idx in range(len(kp0)):
        num_1 = H[0,0]*kp0[idx,0] + H[0,1]*kp0[idx,1] + H[0,2]
        num_2 = H[1,0]*kp0[idx,0] + H[1,1]*kp0[idx,1] + H[1,2]
        denom = H[2,0]*kp0[idx,0] + H[2,1]*kp0[idx,1] + H[2,2]

        J[idx,0] = kp0[idx,0]/denom
        J[idx,1] = kp0[idx,1]/denom
        J[idx,2] = 1/denom
        J[idx,6] = -kp0[idx,0]*num_1/denom
        J[idx,7] = -kp0[idx,1]*num_1/denom
        J[idx,8] = -num_1/denom 
        # now the next row
        J[idx+1,3] = kp0[idx,0]/denom
        J[idx+1,4] = kp0[idx,1]/denom
        J[idx+1,5] = 1/denom
        J[idx+1,6] =  -kp0[idx,0]*num_2/denom
        J[idx+1,7] =  -kp0[idx,1]*num_2/denom
        J[idx+1,8] =  -num_2/denom
    return J

def lm_algorithm(kp_0, kp_1, H_ini, tau=1.5,max_iters=100):
   # our implementation of the LM algorithm
   J_k = jacobian(H_ini, kp_0)
   # initial mu value
   mu_k = tau * np.max(np.diag(J_k.T@J_k))
   # make H be flat instead of a matrix
   H_k = H_ini.ravel()
   C_arr = []
   for iter in range(max_iters):
      # jacobian calculation
      J_f = jacobian(H_k.reshape(3,3), kp_0)
      # error calculation
      eps_pk = error_f(H_k, kp_0, kp_1)
      # cost calculation
      C_pk = np.linalg.norm(eps_pk)**2
      # get delta_p
      delta_p = np.linalg.inv(J_f.T @ J_f + mu_k*np.eye(9)) @ J_f.T @ eps_pk   
      # get new H
      H_new = H_k + delta_p
      # error with new H
      eps_pk1 = error_f(H_new, kp_0, kp_1)
      # new H cost
      C_pk1 = np.linalg.norm(eps_pk1)**2
      C_arr.append([iter,C_pk1])
      # get rho_{k+1}
      rho_n = C_pk - C_pk1
      rho_d = delta_p.T @ (mu_k*np.eye(9) @ delta_p) + delta_p.T @ (J_f.T @ eps_pk)

      rho_k1 = rho_n/rho_d
      # now the quality check
      if rho_n > 0:
         # if rho_n is positive, that means that the new cost is lower than the old
         # which is good, that is what we are looking for
         H_k = H_new
         # pick our new mu_k
         mu_k = mu_k * max(1./3, 1 - (2*rho_k1 - 1)**3)
      else:
         # now, in the case that rho_n is negative, that means our cost has increased
         # so we multiply mu_k by 2 and try again
         mu_k = 2*mu_k
         H_k = H_k

   return H_new.reshape(3,3), np.array(C_arr)

def plot_function(costs,figname):
   # just to plot the cost functions
   plt.clf()
   plt.figure()
   plt.plot(costs[:,0],costs[:,1],color='r')
   plt.xlabel("Iteration")
   plt.ylabel("C(p_k)")
   plt.savefig(figname)
   plt.close()

def get_homography(detector, img0, img1, sigma=2,epsilon=0.1,n_points=10, mode="LM",tau=1.5):
    # this function will get the keypoints for 2 images, run runsac outlier rejection on them,
    # and then get the homography with least squares, which after it will run LM scipy or our implementation
    mkpts0, mkpts1, _ = detector.match(img0, img1)
    inliers_1, inliers_2, outliers_1, outliers_2 = ransac(mkpts0, mkpts1, sigma=sigma, epsilon=epsilon, n_points=n_points)
    # for least squares we can reuse this function that we used in ransac
    H_leastsquares = get_H_matrix_overdetermined(inliers_1, inliers_2, len(inliers_1))
    # we also get the lev-mar optimization lm_algorithm(kp_0, kp_1, H_ini, tau=0.5,max_iters=100,thresh=1e-8)
    if mode == "LS": #LS for least squares / LM for Leverberg Marquadt
        #print("Cost for H_leastsquares: ",np.linalg.norm(error_f(H_leastsquares,inliers_1,inliers_2))**2)
        return inliers_1, inliers_2, outliers_1, outliers_2, H_leastsquares, None
    else:
        # uncomment the following line for own implementation of LM
        H_LM, cost = lm_algorithm(inliers_1, inliers_2, H_leastsquares,tau=tau)
        # the following line is the scipy version of LM
        H_LM_scipy = least_squares(error_f, H_leastsquares.ravel(), method='lm', args=(inliers_1, inliers_2),verbose=False).x.reshape(3,3)
        print("Cost for H_leastsquares: ", np.linalg.norm(error_f(H_leastsquares,inliers_1,inliers_2))**2)
        print("Cost for H_LM: ",np.linalg.norm(error_f(H_LM,inliers_1,inliers_2))**2)
        print("Cost for H_LM scipy: ",np.linalg.norm(error_f(H_LM_scipy,inliers_1,inliers_2))**2)
        print("Change in cost: ", np.linalg.norm(error_f(H_leastsquares,inliers_1,inliers_2))**2 - np.linalg.norm(error_f(H_LM,inliers_1,inliers_2))**2)
        # when using own LM implementation, change the final return value to cost
        # when using scipy LM change it to None
        return inliers_1, inliers_2, outliers_1, outliers_2, H_LM, cost

def plot_inliers_outliers(image1, image2, inliers_1, inliers_2, outliers_1, outliers_2, figname):
    # this is a modified function from superglue_ece661.py, plot_keypoints in particular
    tot_img = np.hstack((image1, image2))
    tot_img = cv2.cvtColor(tot_img, cv2.COLOR_BGR2RGB)
    _, w0, _ = image1.shape
    ms = 0.5
    lw = 0.5
    plt.figure()
    #plt.clf()
    plt.imshow(tot_img);
    # plot inliers in green (match and keypoint)
    for ikp0 in range(len(inliers_1)):
        plt.plot(inliers_1[ikp0,0], inliers_1[ikp0,1], 'g.', markersize=ms)
        plt.plot(inliers_2[ikp0,0]+w0, inliers_2[ikp0,1], 'g.', markersize=ms)         
        plt.plot((inliers_1[ikp0,0], inliers_2[ikp0,0]+w0), (inliers_1[ikp0,1], inliers_2[ikp0,1]), '--gx', linewidth=lw, markersize=ms)
    # plot outliers in red
    for ikp0 in range(len(outliers_1)):
        plt.plot(outliers_1[ikp0,0], outliers_1[ikp0,1], 'r.', markersize=ms)
        plt.plot(outliers_2[ikp0,0]+w0, outliers_2[ikp0,1], 'r.', markersize=ms)
        plt.plot((outliers_1[ikp0,0], outliers_2[ikp0,0]+w0), (outliers_1[ikp0,1], outliers_2[ikp0,1]), '--rx', linewidth=lw, markersize=ms)
    plt.axis('off');
    plt.savefig(figname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return

def create_panorama(left ,middle, right, detector, sigma=2,epsilon=0.1,n_points=8,tau=1.5, mode="LM",task="task1"):
    # for example, we have 5 images numbered 1 through 5
    # the panorama we want looks like this: 1 2 3 4 5
    # so we know 1 and 2 are to the left, 3 is the middle and 4 and 5 are to the right
    # so left, and right are arrays of the form: [1, 2] for left and [4,5] for right
    # since we use superglue and superpoint these have to have the names of the images
    # then, the main idea is that we will map everything to the middle image
    # for this, we get the following homographies:
    # 1 to 2, 2 to 3, then we get the right side ones: 5 to 4 and 4 to 3
    # we do it in this manner since we can just multiply them to find all the homographies 
    # that map everything to 3
    # after getting all the homographies, we can find out the exact shape of our final image by getting the coordinates of all the corners and finding the min and max for both axis
    # to paint all the images to the final canvas, we create binary roi masks, similar to homework 1 to know which image goes where
    # then to conduct less matrix multiplications, we find all the positions where the binary mask is 255, which we then apply
    # homography transformation to them
    # after all this, we can paint all the images to our panorama
    # we will make use of dictionaries to help us out

    image_1_info = {}
    image_2_info = {}
    image_3_info = {}
    image_4_info = {}
    image_5_info = {}

    image_1_info["image"] = cv2.imread(left[0])
    image_2_info["image"] = cv2.imread(left[1])
    image_3_info["image"] = cv2.imread(middle)
    image_4_info["image"] = cv2.imread(right[0])
    image_5_info["image"] = cv2.imread(right[1])

    # 1 to 2
    image_1_info["inliers_1"], image_1_info["inliers_2"], image_1_info["outliers_1"], image_1_info["outliers_2"], image_1_info["homography"], image_1_info["cost"] = get_homography(detector, left[0], left[1], sigma=sigma, epsilon=epsilon, n_points=n_points,mode=mode,tau=tau)
    # 2 to 3
    image_2_info["inliers_1"], image_2_info["inliers_2"], image_2_info["outliers_1"], image_2_info["outliers_2"], image_2_info["tot_homography"], image_2_info["cost"] = get_homography(detector, left[1], middle, sigma=sigma, epsilon=epsilon, n_points=n_points,mode=mode)
    # 4 to 3
    image_4_info["inliers_1"], image_4_info["inliers_2"], image_4_info["outliers_1"], image_4_info["outliers_2"], image_4_info["tot_homography"], image_4_info["cost"] = get_homography(detector, right[0], middle, sigma=sigma, epsilon=epsilon, n_points=n_points, mode=mode)
    # 5 to 4
    image_5_info["inliers_1"], image_5_info["inliers_2"], image_5_info["outliers_1"], image_5_info["outliers_2"], image_5_info["homography"], image_5_info["cost"] = get_homography(detector, right[1], right[0], sigma=sigma, epsilon=epsilon, n_points=n_points,mode=mode)
    # 1 -> 3 homography
    image_1_info["tot_homography"] = image_2_info["tot_homography"] @ image_1_info["homography"]
    # 5 to 3 homography
    image_5_info["tot_homography"] = image_4_info["tot_homography"] @ image_5_info["homography"]
    base_name = task
    #PLOT COST FUNCTIONS:
    if mode == "LM":
        if image_1_info["cost"] is not None:
            plot_function(image_1_info["cost"],base_name+"_LM_cost_image_1.jpg")
        if image_2_info["cost"] is not None:
            plot_function(image_2_info["cost"],base_name+"LM_cost_image_2.jpg")
        if image_4_info["cost"] is not None:
            plot_function(image_4_info["cost"],base_name+"LM_cost_image_4.jpg")
        if image_5_info["cost"] is not None:
            plot_function(image_5_info["cost"],base_name+"LM_cost_image_5.jpg")
    # plot inliers and outliers for different pairs:
    # image 1 to 2
    plot_inliers_outliers(image_1_info["image"], image_2_info["image"], image_1_info["inliers_1"], image_1_info["inliers_2"], image_1_info["outliers_1"], image_1_info["outliers_2"],base_name+ "image_12_inlieroutlier_.jpg")
    # image 2 to 3
    plot_inliers_outliers(image_2_info["image"], image_3_info["image"], image_2_info["inliers_1"], image_2_info["inliers_2"], image_2_info["outliers_1"], image_2_info["outliers_2"],base_name+ "image_23_inlieroutlier_.jpg")
    # image 4 to 3, we swap the orders since we mapped 4 to 3 instead of 3 to 4
    plot_inliers_outliers(image_3_info["image"], image_4_info["image"], image_4_info["inliers_2"], image_4_info["inliers_1"], image_4_info["outliers_2"], image_4_info["outliers_1"],base_name+ "image_34_inlieroutlier_.jpg")
    # image 4 to 5, also swap the orders since we mapped 5 to 4 instead of 4 to 5
    plot_inliers_outliers(image_4_info["image"], image_5_info["image"], image_5_info["inliers_2"], image_5_info["inliers_1"], image_5_info["outliers_2"], image_5_info["outliers_1"],base_name+ "image_45_inlieroutlier_.jpg")


    # we get the corners
    image_1_info["corners"] = get_new_corners(image_1_info["image"],image_1_info["tot_homography"])
    image_2_info["corners"] = get_new_corners(image_2_info["image"],image_2_info["tot_homography"])
    image_4_info["corners"] = get_new_corners(image_4_info["image"],image_4_info["tot_homography"])
    image_5_info["corners"] = get_new_corners(image_5_info["image"],image_5_info["tot_homography"])
    # we just use the identity for the 3 -> 3 homography
    image_3_info["corners"] = get_new_corners(image_3_info["image"], np.eye(3))
    # x min and x max are easy to get since its just the very edge images (1 and 5)
    x_min = int(image_1_info["corners"][:,0].min())
    x_max = int(image_5_info["corners"][:,0].max())
    # y min and max are a bit more involved, since they are all technically at the "same" y, so we need to compare to
    # the y min and y max of all 5 images
    y_min = int(np.array([image_1_info["corners"][:,1].min(),image_2_info["corners"][:,1].min(),image_3_info["corners"][:,1].min(),image_4_info["corners"][:,1].min(),image_5_info["corners"][:,1].min()]).min())
    y_max = int(np.array([image_1_info["corners"][:,1].max(),image_2_info["corners"][:,1].max(),image_3_info["corners"][:,1].min(),image_4_info["corners"][:,1].max(),image_5_info["corners"][:,1].max()]).max())

    resulting_images = {}
    # from both of these we can get the shape of the resulting panorama
    panorama_w = int(x_max - x_min)
    panorama_h = int(y_max - y_min)
    # create the panorama
    panorama = np.zeros((panorama_h,panorama_w,3)).astype(np.uint8)
    # now we just need the binary masks
    image_1_info["binary"] = get_binary_mask(panorama, image_1_info["corners"],x_min, y_min)
    image_2_info["binary"] = get_binary_mask(panorama, image_2_info["corners"],x_min, y_min)
    image_3_info["binary"] = get_binary_mask(panorama, image_3_info["corners"],x_min, y_min)
    image_4_info["binary"] = get_binary_mask(panorama, image_4_info["corners"],x_min, y_min)
    image_5_info["binary"] = get_binary_mask(panorama, image_5_info["corners"],x_min, y_min)
    # get all the coordinates where the binary mask is 255 in a list so we can use broadcasting,
    # we want to do this so we ONLY consider the spots where there is an image and ignore all the rest
    # just less loops to go through when painting the final image
    # but when we do argwhere, the x and y axis are swapped since thats how opencv reads images, so we just swap them back with indexing
    image_1_info["new_coordinates"] = (np.argwhere(image_1_info["binary"] == 255) + np.array([[y_min,x_min]]))[:,[1,0]]
    image_2_info["new_coordinates"] = (np.argwhere(image_2_info["binary"] == 255) + np.array([[y_min,x_min]]))[:,[1,0]]
    image_3_info["new_coordinates"] = (np.argwhere(image_3_info["binary"] == 255) + np.array([[y_min,x_min]]))[:,[1,0]]
    image_4_info["new_coordinates"] = (np.argwhere(image_4_info["binary"] == 255) + np.array([[y_min,x_min]]))[:,[1,0]]
    image_5_info["new_coordinates"] = (np.argwhere(image_5_info["binary"] == 255) + np.array([[y_min,x_min]]))[:,[1,0]]

    # get the converted positions (apply inverse homography, since we are mapping from the range to the domain)
    image_1_info["old_coordinates"] = apply_homography(image_1_info["new_coordinates"],np.linalg.inv(image_1_info["tot_homography"]))
    image_2_info["old_coordinates"] = apply_homography(image_2_info["new_coordinates"],np.linalg.inv(image_2_info["tot_homography"]))
    image_3_info["old_coordinates"] = apply_homography(image_3_info["new_coordinates"],np.linalg.inv(np.eye(3)))
    image_4_info["old_coordinates"] = apply_homography(image_4_info["new_coordinates"],np.linalg.inv(image_4_info["tot_homography"]))
    image_5_info["old_coordinates"] = apply_homography(image_5_info["new_coordinates"],np.linalg.inv(image_5_info["tot_homography"]))
    # with all this information we can now paint everything in our panorama
    panorama = paint_image(panorama, image_1_info["image"], image_1_info["old_coordinates"],image_1_info["new_coordinates"],x_min, y_min)
    panorama = paint_image(panorama, image_2_info["image"], image_2_info["old_coordinates"],image_2_info["new_coordinates"],x_min,y_min)
    panorama = paint_image(panorama, image_3_info["image"], image_3_info["old_coordinates"],image_3_info["new_coordinates"],x_min,y_min)
    panorama = paint_image(panorama, image_4_info["image"], image_4_info["old_coordinates"],image_4_info["new_coordinates"],x_min,y_min)
    panorama = paint_image(panorama, image_5_info["image"], image_5_info["old_coordinates"],image_5_info["new_coordinates"],x_min,y_min)
    # save all the images and intermediate binary masks to show in report
    resulting_images["panorama"] = panorama
    resulting_images["1_binary"] = image_1_info["binary"]
    resulting_images["2_binary"] = image_2_info["binary"]
    resulting_images["3_binary"] = image_3_info["binary"]
    resulting_images["4_binary"] = image_4_info["binary"]
    resulting_images["5_binary"] = image_5_info["binary"]
    return resulting_images

# load superglue+superpoint weights
detector = SuperGlue.create()

# TASK ONE
left = ["ece661_sample_images/1.jpg","ece661_sample_images/2.jpg"]
middle = "ece661_sample_images/3.jpg"
right = ["ece661_sample_images/4.jpg","ece661_sample_images/5.jpg"]
task = "task1_"
# least squares homography panorama
mode = "LS"
im = create_panorama(left ,middle, right, detector, sigma=2,epsilon=0.1,n_points=10, mode=mode,task=task)
cv2.imwrite(task + "panorama_" + mode + ".jpg",im["panorama"])
# own implementation of LM homography panorama
mode = "LM"
im = create_panorama(left ,middle, right, detector, sigma=2,epsilon=0.1,n_points=10, mode=mode,task=task)
cv2.imwrite(task + "panorama_" + mode + ".jpg",im["panorama"])

# TASK TWO
left = ["ece661_sample_images/1_b.jpg","ece661_sample_images/2_b.jpg"]
middle = "ece661_sample_images/3_b.jpg"
right = ["ece661_sample_images/4_b.jpg","ece661_sample_images/5_b.jpg"]
task = "task2_"
# least squares homography panorama
mode = "LS"
im = create_panorama(left ,middle, right, detector, sigma=4,epsilon=0.3,n_points=10, mode="LS",tau=10,task=task)
cv2.imwrite(task + "panorama_" + mode + ".jpg",im["panorama"])
# own implementation of LM panorama
mode = "LM"
im = create_panorama(left ,middle, right, detector, sigma=4,epsilon=0.3,n_points=10, mode="LM",tau=10,task=task)
cv2.imwrite(task + "panorama_" + mode + ".jpg",im["panorama"])
