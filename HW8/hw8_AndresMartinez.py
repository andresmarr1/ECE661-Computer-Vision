
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# we use these to plot the camera poses
from mpl_toolkits.mplot3d.art3d import *
from matplotlib.patches import Rectangle

def get_point_or_line(point1,point2):
    # first we check if these are already in homogeneous form or not, if not we put them in homogeneous representation
    if len(point1) == 2:
        point1 = np.array(point1)
        point1 = np.append(point1, 1)
    if len(point2) == 2:
        point2 = np.array(point2)
        point2 = np.append(point2, 1)
    # then get the cross product
    # this function has uses for us as both getting the line representation and
    # getting line intersections, as the math is the same
    r = np.cross(point1,point2)
    if r[2] != 0:
        return r/r[2]
    return r

def plot_hough_lines(img, lines, name):
    # this is just a function we use to plot the results of the hough line detector, prior to filtering and averaging
    img_ = img.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            img_ = cv2.line(img_, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(name, img_)

def plot_line_rep(line, image, horizontal=False):
    # this so we get the endpoints for the lines to plot after separating into vertical and horizontal
    # notice how we calculate different endpoints depending on if its vertical or not
    # this is because if its horizontal we want the intercepts with the left and rightmost part of the image,
    # and if its vertical we want the same but with top and bottom
    if horizontal:
        ref_1 = get_point_or_line([0,10],[0,15])
        ref_2 = get_point_or_line([image.shape[1]-1,10],[image.shape[1]-1,15])
    else:
        ref_1 = get_point_or_line([10,0],[15,0])
        ref_2 = get_point_or_line([10,image.shape[0]-1],[15,image.shape[0]-1])
    point_1 = get_point_or_line(line, ref_1)[:2]
    point_2 = get_point_or_line(line, ref_2)[:2]
    return point_1, point_2

def cvl_to_homogeneous(cv2_line):
    # we use this to convert the line from rho theta to homogeneous coordinates
    # we operate most of the time with homogeneous as it is easier to use
    rho = cv2_line[0][0]
    theta = cv2_line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    
    return get_point_or_line(pt1, pt2)

def separate_lines(lines):
    # we separate lines by horizontal lines (theta < pi/4 and rho positive, or theta>3pi/4 and negative rho),
    # and if they are not horizontal then they are vertical
    horizontal = []
    vertical = []
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        if (theta <= np.pi/4 and rho >= 0) or (theta > 3*np.pi/4 and rho < 0):
            vertical.append(line)
        else:
            horizontal.append(line)
    return vertical, horizontal

def group_lines(lines, tol=30, image_shape=(600,600), horizontal=False, grid_shape=(8,10)):
    # in here we group lines together with depending on the spot they intersect the edges of the image
    # for horizonal lines we check where it hits the left edge of the image (y=0)
    # for vertical lines we check where it hits the top of the image (x=0)
    # we group them by checking which intersects are within a distance threshold of another interect
    # after we have grouped them together, we 
    if horizontal:
        ref_line = get_point_or_line([0,10],[0,15])
        ref_line_n = get_point_or_line([image_shape[1],10],[image_shape[1],15]) 
        
    else:
        ref_line = get_point_or_line([10,0],[15,0])
        ref_line_n = get_point_or_line([10,image_shape[0]],[15,image_shape[0]]) 
    tot = []
    added = [False] * len(lines)
    for idx,iline in enumerate(lines):
        aux = []
        iline_rep = cvl_to_homogeneous(iline)
        ref_inter = get_point_or_line(iline_rep, ref_line)
        if not added[idx]:
            aux.append(iline_rep)
            added[idx] = True
            for jdx,jline in enumerate(lines):
                if (jline != iline).any():
                    jline_rep = cvl_to_homogeneous(jline)
                    p_inter = get_point_or_line(jline_rep, ref_line)
                    if horizontal:
                        dist = np.abs(ref_inter[1] - p_inter[1])
                    else:
                        dist = np.abs(ref_inter[0] - p_inter[0])
                    if dist < tol and added[jdx] == False:
                        aux.append(jline_rep)
                        added[jdx] = True
            tot.append(aux)     
    # now that we have them separated by groups, we "average" those which have 2 or more lines
    a_lines = []
    for lines in tot:
        if len(lines) > 1:
            inter_point1 = np.array([0,0],dtype=float)
            inter_point2 = np.array([0,0],dtype=float)
            count = 0
            for line in lines:
                inter_point1 += get_point_or_line(ref_line,line)[:2]
                inter_point2 += get_point_or_line(ref_line_n,line)[:2]
                count += 1
            inter_point1 = inter_point1/count
            inter_point2 = inter_point2/count
            n_line = get_point_or_line(inter_point1, inter_point2)
        else:
            n_line = lines[0]
        a_lines.append(n_line)
    

    # now that we have grouped and averaged the lines, we will order them with their x or y intercept depending on the orientation
    n_intercept = []
    for line in a_lines:
        intercept = get_point_or_line(line, ref_line)
        if horizontal:
            n_intercept.append(intercept[1])
        else:
            n_intercept.append(intercept[0])
    # we sort them with their intercepts
    n_intercept = np.array(n_intercept)
    inter_idx_sorted = np.argsort(n_intercept)
    t_lines = []
    for idx in inter_idx_sorted:
        t_lines.append(a_lines[idx])

    # this is the very last step, it will only filter the excess ones, this is mainly used for our dataset 2
    if horizontal:
        if len(t_lines) > grid_shape[1]:
            t_lines = t_lines[:grid_shape[1]]
    else:
        if len(t_lines) > grid_shape[0]:
            t_lines = t_lines[:grid_shape[0]]
    return t_lines

def get_corners(image, tol=15, save_canny=None, save_lines=None, save_hough=None, canny_p1=370, canny_p2=300, hough_p=50):
    # this function runs canny, houghlines, groups the lines, plots all of this,
    # also will loop through all the lines and get the intersects, those are our corners.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,canny_p1,canny_p2)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_p, None, 0, 0)
    if save_hough != None:
        plot_hough_lines(image, lines, save_hough[:-4]+"_houghlines.jpg")
    if save_canny != None:
        cv2.imwrite(save_canny[:-4]+"_canny.jpg", edges)
    
    vert_lines, hor_lines = separate_lines(lines)
    v_lines = group_lines(vert_lines,tol=tol, image_shape=(image.shape[1],image.shape[0]), horizontal=False)
    h_lines = group_lines(hor_lines,tol=tol, image_shape=(image.shape[1],image.shape[0]), horizontal=True)
    if save_lines != None:
        line_plot = image.copy()
        for vline in v_lines:
            p1, p2 = plot_line_rep(vline, line_plot, horizontal=False)
            p1_ = (int(p1[0]), int(p1[1]))
            p2_ = (int(p2[0]), int(p2[1]))
            line_plot = cv2.line(line_plot,p1_, p2_, (255,0,0), 2);
        for hline in h_lines:
            p1, p2 = plot_line_rep(hline, line_plot, horizontal=True)
            p1_ = (int(p1[0]), int(p1[1]))
            p2_ = (int(p2[0]), int(p2[1]))
            line_plot = cv2.line(line_plot,p1_, p2_, (0,255,0), 2);
        cv2.imwrite(save_lines[:-4]+"_fillines.jpg", line_plot)
        
    corners = []
    #this is how we get the corners
    for h_line in h_lines:
        for v_line in v_lines:
            corner = get_point_or_line(h_line, v_line)[:2]
            corners.append(corner)
    return corners

# reusing these from my hw5
def point_to_point_system(domain, range_,num_points):
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

def linear_least_squares(mat1, mat2):
    # we make a distinction because this is for inhomogeneous least squares
    return np.linalg.inv(mat1.T @ mat1) @ mat1.T @ mat2

def linear_least_squares_homogeneous(mat1):
    # and this is for homogeneous least squares
    _, _, v_t = np.linalg.svd(mat1.T @ mat1)
    return v_t[-1]

def get_H_matrix(domain,range_):
    # I reuse this from hw5
    n_points = len(domain)
    mat1, mat2 = point_to_point_system(domain, range_,n_points)
    # we use this function to solve the equation described in the logic, 
    # I replaced np.dot with @ as that is what they suggest in the numpy website
    # in this case we use the pseudo inverse (ATA)^-1 AT
    sol = linear_least_squares(mat1,mat2)
    # we append the 1 since this will only have 8 values, the 1 is missing
    sol = np.append(sol,np.array([[1]]),axis=0)
    return sol.reshape((3,3)).astype(float)

def add_ones(array):
    # this function takes in an array and adds ones
    # this is mainly for applying homographies
    ones_np = np.ones(array.shape[:-1])[...,None]
    # need to expand the dimensions by 1 to be able to concatenate it
    return np.append(array,ones_np,axis=-1)
def add_number(array,n):
    # this function takes in an array and adds any number
    # this is mainly for adding a 0, for z=0 in the real life plane coordinates
    ones_np = np.ones(array.shape[:-1])[...,None]
    ones_np *= n
    # need to expand the dimensions by 1 to be able to concatenate it
    return np.append(array,ones_np,axis=-1)

def apply_homography(positions, H):
    # reused from hw5
    # this gets the homography transformation for all the coordinates in the image that we get from the get_positions function
    # we do it in a way that exploits broadcasting, so we don't need to use for loops
    temp_pos = add_ones(positions)
    new_pos = (H @ temp_pos.T).astype(float)
    new_pos /= new_pos[2,:]
    return new_pos[:2,:].T

def get_coordinates(shape=(8,10),length=5):
    # this is how we create our 2d grid that is on Z=0,
    # for both datasets we end up using length =1 since the squares are of 1 inch length
    x = np.linspace(0, length*(shape[0] - 1), shape[0])
    y = np.linspace(0, length*(shape[1] - 1), shape[1])
    xv, yv = np.meshgrid(x,y)
    xv = xv.ravel()[:, None]
    yv = yv.ravel()[:, None]
    return np.hstack((xv,yv))

def get_vij(H, i,j):
    # we change ij to go from 0 to 2 rather than 1 to 3, to adjust for python indexes
    # we follow the equations from the scrolls
    v_ij = np.array([H[0,i]*H[0,j], 
                     H[0,i]*H[1,j]+H[0,j]*H[1,i],
                     H[1,i]*H[1,j],
                     H[0,i]*H[2,j]+H[0,j]*H[2,i],
                     H[1,i]*H[2,j]+H[1,j]*H[2,i],
                     H[2,i]*H[2,j]])
    return v_ij

def get_V(H):
    # this is to get the V matrix
    v_11 = get_vij(H,0,0)[..., None]
    v_12 = get_vij(H,0,1)[..., None]
    v_22 = get_vij(H,1,1)[..., None]
    V = np.vstack((v_12.T,(v_11 - v_22).T))
    return V

def get_b(V):
    # we get b from the stack of Vs
    b = linear_least_squares_homogeneous(V)
    return b

def get_omega(V):
    # this is the entire stack of Vs,
    # we get omega from b which we get from the stack of Vs
    b = get_b(V)
    omega = np.array([[b[0], b[1], b[3]],
                      [b[1], b[2], b[4]],
                      [b[3], b[4], b[5]]])
    return omega

def get_K_matrix(omega):
    # we follow the instructions from the scrolls and build our K
    
    x_0 = (omega[0,1]*omega[0,2] - omega[0,0]*omega[1,2])/(omega[0,0]*omega[1,1] - omega[0,1]**2)
    lambda_ = omega[2,2] - (omega[0,2]**2 + x_0*(omega[0,1]*omega[0,2] - omega[0,0]*omega[1,2]))/(omega[0,0])
    alpha_x = np.sqrt(lambda_/omega[0,0])
    alpha_y = np.sqrt(lambda_*omega[0,0]/(omega[0,0]*omega[1,1] - omega[0,1]**2))
    s = -1 * (omega[0,1] * alpha_x**2 * alpha_y)/(lambda_)
    y_0 = (s * x_0 / alpha_y) - (omega[0,2]*alpha_x**2)/(lambda_)
    K = np.array([[alpha_x, s, x_0],
                  [0, alpha_y, y_0],
                  [0, 0, 1]])
    return K

def get_extrinsics(K, H):
    # just following the instructions from the scrolls
    scale = 1 / np.linalg.norm(np.linalg.inv(K) @ H[:,0])
    r_1 = scale * np.linalg.inv(K) @ H[:,0]
    r_2 = scale * np.linalg.inv(K) @ H[:,1]
    r_3 = np.cross(r_1, r_2)
    t = scale * np.linalg.inv(K) @ H[:,2]
    R = np.column_stack((r_1,r_2,r_3))
    # now we condition R, very important step
    u, _, v = np.linalg.svd(R)
    R_conditioned = u @ v
    return R_conditioned, t

def get_rodrigues_rep_from_R(R):
    # we get the rodrigues vector from the rotation matrix, again following the scrolls
    phi = np.arccos((np.trace(R)-1)/2) 
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]])
    
    return w * (phi/(2*np.sin(phi)))

def get_R_from_rodrigues(w_0, w_1, w_2):
    # this function is to help us reconstruct our matrices during LM optimization,
    # more on that in the following functions
    # again we follow the scrolls
    W_X = np.array([[0, -w_2, w_1],
                    [w_2, 0, -w_0],
                    [-w_1, w_0, 0]])
    phi = np.sqrt(w_0**2 + w_1**2 + w_2**2)
    
    R = np.eye(3) + (np.sin(phi)/phi) * W_X + ((1 - np.cos(phi))/(phi**2)) * W_X @ W_X
    return R

def get_params_from_matrices(K, R_list, t_list):
    params = []
    # get the parameters needed from K, R, and t
    # R and t are per image,
    # since K is shared between all images, we make sure to optimize it in this way, that is why it is only appended once
    params.append(K[0,0])
    params.append(K[0,1])
    params.append(K[0,2])
    params.append(K[1,1])
    params.append(K[1,2])
    for idx in range(len(R_list)):
        R = R_list[idx]
        t = t_list[idx]
        #we need to use rodrigues instead of the rotation matrix
        w = get_rodrigues_rep_from_R(R)
        params.append(w[0])
        params.append(w[1])
        params.append(w[2])
        params.append(t[0])
        params.append(t[1])
        params.append(t[2])
    return np.array(params)

def get_matrices_from_params(params):
    # this is how we reconstruct our matrices using the parameters
    alpha_x = params[0] 
    alpha_y = params[1]
    s = params[2]
    x_0 = params[3] 
    y_0 = params[4]
    new_R_list = []
    new_t_list = []
    for idx in range(5,len(params),6):
        w_0 = params[idx]
        w_1 = params[idx + 1]
        w_2 = params[idx + 2]
        # we get our R back
        R_ = get_R_from_rodrigues(w_0, w_1, w_2)

        t_0 = params[idx + 3]
        t_1 = params[idx + 4]
        t_2 = params[idx + 5]
        t_ = np.array([t_0, t_1, t_2])

        new_R_list.append(R_)
        new_t_list.append(t_)
    
    K = np.array([[alpha_x, s, x_0],
                  [0, alpha_y, y_0],
                  [0, 0, 1]])

    return K, new_R_list, new_t_list

def get_projection_matrix(K, R, t):
    # this is just following the scrolls, this is the definition of projection matrix
    R_t = np.hstack((R, t[:,None]))
    P = K @ R_t
    return P

def error_f(params,pts_3d, pts_img_list):
    # this is the function we use to optimize,
    # first we get every parameter in a numpy array
    K, R_list, t_list = get_matrices_from_params(params)
    error = np.empty((0),dtype=np.float32)
    # get the error X - f, per image and append it to an error list
    for idx in range(len(R_list)):
        P = get_projection_matrix(K, R_list[idx], t_list[idx])
        reprojected_pts = apply_homography(pts_3d, P)
        # always the geometric error here
        err = np.abs((pts_img_list[idx] - reprojected_pts)).ravel()
        
        error = np.append(error, err, axis=0)
        
    return error

def reprojection_error(pts_3d, pts_img, P):
    # this is just another function to use to give us the same thing but this is just for one image
    reprojected_pts = apply_homography(pts_3d, P)
    return np.abs((pts_img - reprojected_pts)).ravel()

def lm_optim_P(pts_3d, pts_img_list, K, R_list, t_list):
    # this is our optimization function, we call scipy least squares here after getting all the parameters
    params = get_params_from_matrices(K, R_list, t_list)
    op_params = least_squares(error_f, params, method='lm', args=(pts_3d, pts_img_list),verbose=False).x
    # we reconstruct them and this is what we return
    K_op, R_op_list, t_op_list = get_matrices_from_params(op_params)
    
    return K_op, R_op_list, t_op_list

def get_camera_poses(R, t):
    # this is how we get our camera poses,
    # there is one modification we did, we are no longer adding the C into the directions,
    # as the function we call to draw the vectors in 3d gets the direction not the endpoint
    C = - R.T @ t

    X_xcam = np.array([1,0,0])
    X_ycam = np.array([0,1,0])
    X_zcam = np.array([0,0,1])

    X_x = R.T @ X_xcam 
    X_y = R.T @ X_ycam
    X_z = R.T @ X_zcam


    return C, X_x, X_y, X_z

def plot_corners(image, corners_, name):
    # this is just to plot corners and the label
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.copy()
    cont = 0
    ms = 2
    plt.figure()
    plt.imshow(img);
    for corner in corners_:
        x = corner[0]
        y = corner[1]
        plt.plot(x,y, 'g.', markersize=ms)
        plt.text(x, y, str(cont), fontsize=8, color='g')
        cont += 1
    plt.axis('off');
    plt.savefig(name[:-4] + ".png",bbox_inches='tight',pad_inches=None);
    plt.close()
    plt.cla()
    plt.clf()

def plot_corners_improvement(image, corners_, corners_improved, name):
    # we use this to plot the detected corners and the reprojected corners
    # projected corners always in red and detected corners in green
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.copy()
    cont = 0
    ms = 2
    plt.figure()
    plt.imshow(img);
    for corner in corners_:
        x = corner[0]
        y = corner[1]
        plt.plot(x,y, 'r.', markersize=ms)
        #plt.text(x, y, str(cont), fontsize=8, color='g')
        #cont += 1
    for corner in corners_improved:
        x = corner[0]
        y = corner[1]
        plt.plot(x,y, 'g.', markersize=ms)
        plt.text(x, y, str(cont), fontsize=8, color='g')
        cont += 1
    plt.axis('off');
    plt.savefig(name[:-4] + ".png",bbox_inches='tight',pad_inches=None);
    plt.close()
    plt.cla()
    plt.clf()

def reproject_to_fixed_image(fixed_img, fixed_image_data, other_image_list):
    #this is the function we use to call the reprojection into the fixed image, 
    # we are only interested in the first, second and last column of the projection matrix,
    # so we take just that from the projection matrix and use the resulting 3 by 3 matrix to project the points
    P_fixed = fixed_image_data["projection_matrix"]
    H_fixed = P_fixed[:,[0,1,3]]
    corners_fixed = fixed_image_data["corners"]
    P_fixed_op = fixed_image_data["projection_matrix_new"]
    H_fixed_op = P_fixed_op[:,[0,1,3]]
    save_path = "improvement"
    for image_data in other_image_list:
        print("-"*10)
        print(image_data["filename"])
        corners = image_data["corners"]
        P_old = image_data["projection_matrix"]
        H_old = P_old[:,[0,1,3]]

        projection_3d_old = apply_homography(corners, np.linalg.inv(H_old))
        reprojected_fixedim_corners_old = apply_homography(projection_3d_old, H_fixed)

        plot_corners_improvement(fixed_img, reprojected_fixedim_corners_old, corners_fixed, os.path.join(save_path,image_data["filename"][:-4]+"_beforeLM.jpg"))

        P_op = image_data["projection_matrix_new"]
        H_op = P_op[:,[0,1,3]]

        projection_3d_op = apply_homography(corners, np.linalg.inv(H_op))
        reprojected_fixedim_corners_new = apply_homography(projection_3d_op, H_fixed_op)
        plot_corners_improvement(fixed_img, reprojected_fixedim_corners_new, corners_fixed, os.path.join(save_path,image_data["filename"][:-4]+"_afterLM.jpg"))
        err_old = np.linalg.norm(reprojected_fixedim_corners_old - fixed_image_data["corners"],axis=1)

        err_new = np.linalg.norm(reprojected_fixedim_corners_new - fixed_image_data["corners"],axis=1)
        
        print("Mean: ", err_old.mean())
        print("Mean after LM: ", err_new.mean())
        print("-"*4)
        print("Std: ", err_old.std())
        print("Std after LM: ", err_new.std())

def plot_camera_poses(data_dict_list,pattern_shape=(8,10),pattern_size=1.,name=None):
    # this is how we plot the camera poses, it takes all the data at once and loops through it
    plt.close()
    plt.cla()
    plt.clf()
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(projection='3d')
    # this is for the calibration pattern
    for j in range(pattern_shape[1]-1):
        for i in range(pattern_shape[0]-1):
            # need to reflect the fact that it "skips" a line
            if j % 2 == 0:
                # alternate black and white
                color = 'black' if (i + j) % 2 == 0 else 'white'
                x = i * pattern_size
                y = j * pattern_size
                rect = Rectangle((x, y), pattern_size, pattern_size, color=color, alpha=1, edgecolor=None)
                ax.add_patch(rect)
                # Set the 3D position of each rectangle at Z=0
                pathpatch_2d_to_3d(rect, z=0, zdir="z")
            else:
                # in this case, it is all white squares
                color = 'white'
                x = i * pattern_size
                y = j * pattern_size
                rect = Rectangle((x, y), pattern_size, pattern_size, color=color, alpha=1, edgecolor=None)
                ax.add_patch(rect)
                # Set the 3D position of each rectangle at Z=0
                pathpatch_2d_to_3d(rect, z=0, zdir="z")
    # nowe we draw our poses
    for img_data_dict in data_dict_list:
        # loop through the data, choose a random color, the way matplotlib does it is with 0 to 1
        color_random = np.random.random(3)
        center = img_data_dict["camera_pose"]["center"]
        c_x, c_y, c_z = center
        x_dir = img_data_dict["camera_pose"]["x_vector"]
        x_x, x_y, x_z = x_dir
        y_dir = img_data_dict["camera_pose"]["y_vector"]
        y_x, y_y, y_z = y_dir
        z_dir = img_data_dict["camera_pose"]["z_vector"]
        z_x, z_y, z_z = z_dir

        # this is for our direction vectors
        ax.quiver(c_x, c_y, c_z, x_x, x_y, x_z, color='r', linewidth=1)
        ax.quiver(c_x, c_y, c_z, y_x, y_y, y_z, color='g', linewidth=1)
        ax.quiver(c_x, c_y, c_z, z_x, z_y, z_z, color='b', linewidth=1)

        # now the camera plane, in here we notice that the corners are made up from addition and substractions of the camera center and the directions,
        # so we make use to that
        points = [center + x_dir + y_dir,
                center + x_dir - y_dir,
                center - x_dir - y_dir,
                center - x_dir + y_dir]
            
        poly = Poly3DCollection([points], color=color_random, alpha=0.4, edgecolor=None)
        ax.add_collection3d(poly)

    #ax.view_init(45,-90) 
    # we set reasonable limits and plot it
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if name:
        plt.savefig("camera_poses_.pdf",bbox_inches='tight',pad_inches=None);
    else:
        plt.savefig("camera_poses.pdf",bbox_inches='tight',pad_inches=None);
    plt.close()
    plt.cla()
    plt.clf()

img_path = "HW8-Files/Dataset1"
save_path = "imgs"
save_canny = "canny"
save_lines = "lines"
# we get all the files in the dataset to loop through
file_list = [x for x in os.listdir(img_path) if x.endswith(".jpg")]
imgs_data = []
H_list = []
corners_list = []
Vs = []
# again, 1 for length as we are using inches
RL_coords = get_coordinates(shape=(8,10),length=1)
# this is now our 3d coordinate list
RL_3d = add_number(RL_coords,0)

for file in file_list:
    img_data = {}
    img_data["filename"] = file
    img = cv2.imread(os.path.join(img_path,file))
    sname = os.path.join(save_path, file)
    canny_name = os.path.join(save_canny, file)
    line_name = os.path.join(save_lines, file)
    hough_name = os.path.join('houghlines', file)
    corners = get_corners(img, tol=15, save_canny=canny_name, save_lines=line_name, save_hough=hough_name)
    corners_list.append(corners)
    img_data["corners"] = np.array(corners)  
    
    
    plot_corners(img, corners, sname);

    H = get_H_matrix(RL_coords, np.array(corners))
    img_data["homography"] = H
    #print(H)
    if len(corners) == 80:
        # need to make sure that the number of corners is 80
        H_list.append(H)
        
        V_h = get_V(H)
        Vs.append(V_h[0])
        Vs.append(V_h[1])

        imgs_data.append(img_data)
    else:
        print("Incorrect number of lines for ", file)
    
Vs = np.array(Vs)
omega = get_omega(Vs)
K = get_K_matrix(omega)

R_list = []
t_list = []

for idx in range(len(imgs_data)):
    R, t = get_extrinsics(K, imgs_data[idx]["homography"])
    imgs_data[idx]["rotation_matrix"] = R
    imgs_data[idx]["translation_vector"] = t
    imgs_data[idx]["projection_matrix"] = get_projection_matrix(K, R, t)
    R_list.append(R)
    t_list.append(t)
# now the optimization
K_op, R_op_list, t_op_list = lm_optim_P(RL_3d, corners_list, K, R_list, t_list)

for idx in range(len(imgs_data)):
    # loop through the data and populate what is missing
    imgs_data[idx]["rotation_matrix"] = R_list[idx]
    imgs_data[idx]["translation_vector"] = t_list[idx]
    imgs_data[idx]["projection_matrix"] = get_projection_matrix(K, R_list[idx], t_list[idx])

    imgs_data[idx]["K_op"] = K_op 
    imgs_data[idx]["R_op"] = R_op_list[idx] 
    imgs_data[idx]["t_op"] = t_op_list[idx]  

    P_new = get_projection_matrix(K_op, R_op_list[idx], t_op_list[idx])
    P_old = get_projection_matrix(K, R_list[idx], t_list[idx])
    imgs_data[idx]["projection_matrix_new"] = P_new

    imgs_data[idx]["old_reprojection_error"] = reprojection_error(RL_3d, imgs_data[idx]["corners"], P_old)
    imgs_data[idx]["new_reprojection_error"] = reprojection_error(RL_3d, imgs_data[idx]["corners"], P_new)
    imgs_data[idx]["err_mean_old"] = imgs_data[idx]["old_reprojection_error"].mean()
    imgs_data[idx]["err_mean_new"] = imgs_data[idx]["new_reprojection_error"].mean()
    imgs_data[idx]["err_std_old"] = imgs_data[idx]["old_reprojection_error"].std()
    imgs_data[idx]["err_std_new"] = imgs_data[idx]["new_reprojection_error"].std()

    C, X_x, X_y, X_z = get_camera_poses(R_op_list[idx], t_op_list[idx])
    camera_pose = {}
    camera_pose["center"] = C
    camera_pose["x_vector"] = X_x
    camera_pose["y_vector"] = X_y
    camera_pose["z_vector"] = X_z
    imgs_data[idx]["camera_pose"] = camera_pose

print("K before LM: ", K)
print("K after LM: ", K_op)

imgs_to_show = ["Pic_19.jpg","Pic_31.jpg"]
fixed_img = "Pic_11.jpg" 
fixed_image = cv2.imread(os.path.join(img_path, fixed_img))
imgs_show_data = []
for img_data in imgs_data:
    if img_data["filename"] == fixed_img:
        fixed_img_data = img_data
    if img_data["filename"] in imgs_to_show:
        imgs_show_data.append(img_data)

reproject_to_fixed_image(fixed_image, fixed_img_data, imgs_show_data)

for img_show_data in imgs_show_data:
    print("-"*10)
    print(img_show_data["filename"])
    print("Before LM, rotation matrix: ",img_show_data["rotation_matrix"])
    print("Before LM, translation vector: ",img_show_data["translation_vector"])
    print("After LM, rotation matrix: ",img_show_data["R_op"])
    print("After LM, translation vector: ",img_show_data["t_op"])

    img = cv2.imread(os.path.join(img_path, img_show_data['filename']))
    new_points = apply_homography(RL_3d, img_show_data["projection_matrix"])
    new_points_improved = apply_homography(RL_3d, img_show_data["projection_matrix_new"])
    plot_corners_improvement(img, new_points, img_show_data["corners"], os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_beforeLM.jpg"))
    plot_corners_improvement(img, new_points_improved, img_show_data["corners"], os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_afterLM.jpg"))
    plot_corners_improvement(img, new_points, new_points_improved, os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_comparison.jpg"))
    
plot_camera_poses(imgs_data)

# for the second dataset we follow exactly the same steps
img_path_ = "dataset2/small"
save_path = "imgs"
save_canny = "canny"
save_lines = "lines"
file_list_ = [x for x in os.listdir(img_path_) if x.endswith(".jpeg")]
imgs_data_ = []
H_list_ = []
corners_list_ = []
Vs_ = []
RL_coords = get_coordinates(shape=(8,10),length=1)
RL_3d = add_number(RL_coords,0)

for file in file_list_:
    img_data_ = {}
    img_data_["filename"] = file
    img = cv2.imread(os.path.join(img_path_,file))
    sname = os.path.join(save_path, file)
    canny_name = os.path.join(save_canny, file)
    line_name = os.path.join(save_lines, file)
    hough_name = os.path.join('houghlines', file)
    corners = get_corners(img, tol=20, save_canny=canny_name, save_lines=line_name, save_hough=hough_name, canny_p1=370, canny_p2=300, hough_p=50)
    corners_list_.append(corners)
    img_data_["corners"] = np.array(corners)  
    
    if len(corners) == 80:
        
        plot_corners(img, corners, sname);

        H = get_H_matrix(RL_coords, np.array(corners))
        img_data_["homography"] = H
        #print(H)
        H_list_.append(H)
        
        V_h = get_V(H)
        Vs_.append(V_h[0])
        Vs_.append(V_h[1])

        imgs_data_.append(img_data_)
    else:
        print(file)
        print(len(corners))
    
Vs_ = np.array(Vs_)
omega = get_omega(Vs_)
K_ = get_K_matrix(omega)

R_list_ = []
t_list_ = []

for idx in range(len(imgs_data_)):
    R, t = get_extrinsics(K_, imgs_data_[idx]["homography"])
    imgs_data_[idx]["rotation_matrix"] = R
    imgs_data_[idx]["translation_vector"] = t
    imgs_data_[idx]["projection_matrix"] = get_projection_matrix(K_, R, t)
    R_list_.append(R)
    t_list_.append(t)
    
K_op_, R_op_list_, t_op_list_ = lm_optim_P(RL_3d, corners_list_, K_, R_list_, t_list_)

for idx in range(len(imgs_data_)):
    imgs_data_[idx]["rotation_matrix"] = R_list_[idx]
    imgs_data_[idx]["translation_vector"] = t_list_[idx]
    imgs_data_[idx]["projection_matrix"] = get_projection_matrix(K_, R_list_[idx], t_list_[idx])

    imgs_data_[idx]["K_op"] = K_op_ 
    imgs_data_[idx]["R_op"] = R_op_list_[idx] 
    imgs_data_[idx]["t_op"] = t_op_list_[idx]  

    P_new = get_projection_matrix(K_op_, R_op_list_[idx], t_op_list_[idx])
    P_old = get_projection_matrix(K_, R_list_[idx], t_list_[idx])
    imgs_data_[idx]["projection_matrix_new"] = P_new

    imgs_data_[idx]["old_reprojection_error"] = reprojection_error(RL_3d, imgs_data_[idx]["corners"], P_old)
    imgs_data_[idx]["new_reprojection_error"] = reprojection_error(RL_3d, imgs_data_[idx]["corners"], P_new)
    imgs_data_[idx]["err_mean_old"] = imgs_data_[idx]["old_reprojection_error"].mean()
    imgs_data_[idx]["err_mean_new"] = imgs_data_[idx]["new_reprojection_error"].mean()
    imgs_data_[idx]["err_std_old"] = imgs_data_[idx]["old_reprojection_error"].std()
    imgs_data_[idx]["err_std_new"] = imgs_data_[idx]["new_reprojection_error"].std()

    C, X_x, X_y, X_z = get_camera_poses(R_op_list_[idx], t_op_list_[idx])
    camera_pose = {}
    camera_pose["center"] = C
    camera_pose["x_vector"] = X_x
    camera_pose["y_vector"] = X_y
    camera_pose["z_vector"] = X_z
    imgs_data_[idx]["camera_pose"] = camera_pose

print("K before LM: ", K_)
print("K after LM: ", K_op_)

imgs_to_show = ["pose9.jpeg","pose33.jpeg"]
fixed_img = "fixed_img.jpeg" 
fixed_image = cv2.imread(os.path.join(img_path_, fixed_img))
imgs_show_data = []
for img_data in imgs_data_:
    if img_data["filename"] == fixed_img:
        fixed_img_data = img_data
    if img_data["filename"] in imgs_to_show:
        imgs_show_data.append(img_data)

reproject_to_fixed_image(fixed_image, fixed_img_data, imgs_show_data)

print(fixed_img_data["rotation_matrix"])
print(fixed_img_data["translation_vector"])
print(fixed_img_data["R_op"])
print(fixed_img_data["t_op"])

for img_show_data in imgs_show_data:
    print("-"*10)
    print(img_show_data["filename"])
    print("Before LM, rotation matrix: ",img_show_data["rotation_matrix"])
    print("Before LM, translation vector: ",img_show_data["translation_vector"])
    print("After LM, rotation matrix: ",img_show_data["R_op"])
    print("After LM, translation vector: ",img_show_data["t_op"])

    img = cv2.imread(os.path.join(img_path_, img_show_data['filename']))
    new_points = apply_homography(RL_3d, img_show_data["projection_matrix"])
    new_points_improved = apply_homography(RL_3d, img_show_data["projection_matrix_new"])
    plot_corners_improvement(img, new_points, img_show_data["corners"], os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_beforeLM.jpg"))
    plot_corners_improvement(img, new_points_improved, img_show_data["corners"], os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_afterLM.jpg"))
    plot_corners_improvement(img, new_points, new_points_improved, os.path.join("improvement",img_show_data["filename"][:-4]+"_corners_comparison.jpg"))

plot_camera_poses(imgs_data_,name=True)


