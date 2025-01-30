
import cv2
import matplotlib.pyplot as plt
import numpy as np

def point_to_point_system(domain, range_):
    # requires both dictionaries to have PQRS in [x,y] format
    # this will fill the matrices used to find H

    mat1 = np.empty((0,8),dtype=float)
    mat2 = np.empty((0,0),dtype=float)
    for i in range(domain.shape[0]):

        x = domain[i,0]
        x_prime = range_[i,0]
        y = domain[i,1]
        y_prime = range_[i,1]

        mat1 = np.append(mat1,np.array([[x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime]]),axis=0)
        mat1 = np.append(mat1,np.array([[0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]]),axis=0)
        mat2 = np.append(mat2,x_prime)
        mat2 = np.append(mat2,y_prime)

    return mat1,mat2.reshape(8,1)

def calculate_H_matrix(domain,range):
    mat1, mat2 = point_to_point_system(domain, range)
    # we use this function to solve the equation described in the logic, 
    # I replaced np.dot with @ as that is what they suggest in the numpy website
    sol = np.linalg.inv(mat1) @ mat2
    # we append the 1 since this will only have 8 values, the 1 is missing
    sol = np.append(sol,np.array([[1]]),axis=0)
    return sol.reshape((3,3)).astype(float)

def homography_transform(H, point):
    # this function maps coordinates (x,y) to (x',y') using the homography H
    #p_prime = np.array([point[0],point[1],1],)
    p = np.append(np.array(point),[1]).astype(int)
    prime_p = H @ p 
    return (prime_p/prime_p[2]).astype(int)

def apply_homography(image, H):
    # this function transforms the entire image according to homography matrix H
    # I print out the corners, since our transformation may make the image "go out of its borders"
    # I created another function to handle such cases, and that is the one we mostly use.
    corners = find_corners(image,H)
    print(corners)
    H_inv = np.linalg.inv(H)
    canvas = np.zeros_like(image)
    for i in range(canvas.shape[1]):
        for j in range(canvas.shape[0]):
            pos = [i,j]
            tpos = homography_transform(H_inv,pos)
            if 0 < tpos[0] < canvas.shape[1] and 0 < tpos[1] < canvas.shape[0]:
                canvas[j,i] = image[tpos[1],tpos[0]]
    return canvas

def find_corners(image,H, mod=False):
  #need to find these corners to be able to resize the image
  # I found a strange interaction in some cases, where the x' values should be negative, 
  # so I handle those with mod=True/False, this mainly happens with the corridor image
  p = [0,0]
  q = [image.shape[1],0]
  r = [image.shape[1],image.shape[0]]
  s = [0,image.shape[0]]

  p_p = homography_transform(H,p)
  q_p = homography_transform(H,q)
  r_p = homography_transform(H,r)
  s_p = homography_transform(H,s)
  if mod:
    p_p = -p_p
    s_p = -s_p
  points_prime = np.array([p_p,q_p,r_p,s_p])
  corn = {}
  corn["xmin"] = np.min(points_prime[:,0])
  corn["xmax"] = np.max(points_prime[:,0])
  corn["ymin"] = np.min(points_prime[:,1])
  corn["ymax"] = np.max(points_prime[:,1])
  return corn

def apply_homography_resize(image, H, mod=False):
    # this function transforms the entire image according to homography matrix H
    # this is the function that resizes the image depending on where the corners are
    # need to add the top left (xmin,ymin) to each position before transforming it
    corners = find_corners(image,H,mod=mod)
    print(corners)
    w = int(corners["xmax"] - corners["xmin"])
    h = int(corners["ymax"] - corners["ymin"])
    canvas = np.zeros((h,w,3)).astype(np.uint8)
    H_inv = np.linalg.inv(H)
    for i in range(canvas.shape[1]):
        for j in range(canvas.shape[0]):
            pos = [int(i+corners["xmin"]),int(j+corners["ymin"])]
            tpos = homography_transform(H_inv,pos)
            if 0 <= tpos[0] < image.shape[1] and 0 <= tpos[1] < image.shape[0]:
                canvas[j,i] = image[tpos[1],tpos[0]]
    return canvas

def get_point_or_line(point1, point2):
    #take in 2d or 3d vectors and get their cross product, we end up dividing the entire thing by the x_3
    
    if len(point1) == 2 and len(point2)== 2:
        point1 = np.array(point1)
        point2 = np.array(point2)
        point1 = np.append(point1,1)
        point2 = np.append(point2,1)
    r = np.cross(point1,point2)
    return r/r[2]

##### TWO STEP

def get_vanishing_line(arr):
    #we use the pairs of, what we know are orthogonal in the scene, lines: PQ-RS, PS-QR
    
    #PQ
    line_PQ = get_point_or_line(arr[0],arr[1])
    #RS
    line_RS = get_point_or_line(arr[2],arr[3])
    #PS
    line_PS = get_point_or_line(arr[0],arr[3])
    #QR
    line_QR = get_point_or_line(arr[1],arr[2])

    vpoint1 = get_point_or_line(line_PQ, line_RS)

    vpoint2 = get_point_or_line(line_PS, line_QR)
    
    return get_point_or_line(vpoint1,vpoint2)

def remove_vanishing_line(arr):
    # we use this to calculate the H matrix that sends the vanishing line to the line at infinity
    vanishing_line = get_vanishing_line(arr)
    H_vl = np.eye(3)
    H_vl[2,0] = vanishing_line[0]
    H_vl[2,1] = vanishing_line[1]
    return H_vl

def two_step_system(arr):
    #PQ
    line_PQ = get_point_or_line(arr[0],arr[1])
    #QR
    line_QR = get_point_or_line(arr[1],arr[2])
    #RS
    line_RS = get_point_or_line(arr[2],arr[3])
    #PS
    line_PS = get_point_or_line(arr[0],arr[3])
    
    #following the linear system from the logic
    mat1 = np.empty((0,2),dtype=float)
    mat2 = np.empty((0,0),dtype=float)
    mat1 = np.append(mat1,np.array([[line_PQ[0] * line_QR[0], line_PQ[0] * line_QR[1] + line_PQ[1] * line_QR[0]]]),axis=0)
    mat1 = np.append(mat1,np.array([[line_RS[0] * line_PS[0], line_RS[0] * line_PS[1] + line_RS[1] * line_PS[0]]]),axis=0)
    mat2 = np.append(mat2,np.array([-line_PQ[1] * line_QR[1]]))
    mat2 = np.append(mat2,np.array([-line_RS[1] * line_PS[1]]))
    
    return mat1, mat2.reshape((2,1))

def remove_affine_distortion(arr):
    # follows from the logic
    mat1, mat2 = two_step_system(arr)
    res = np.linalg.inv(mat1)@mat2
    S = np.eye(2)
    S[0,0] = res[0]
    S[0,1] = res[1]
    S[1,0] = S[0,1]

    V, D, _ = np.linalg.svd(S)
    A_eigenvalues = np.sqrt(np.diag(D))
    A = V @ A_eigenvalues @ V.transpose()
    H = np.eye(3,3)
    H[:2,:2] = A

    return np.linalg.inv(H)


##### TWO STEP

def orthogonal_lines(arr):
    #orthogonal pair 1
    line_11 = get_point_or_line(arr[0][0],arr[0][1])
    line_12 = get_point_or_line(arr[0][2],arr[0][3])
    #orthogonal pair 2
    line_21 = get_point_or_line(arr[1][0],arr[1][1])
    line_22 = get_point_or_line(arr[1][2],arr[1][3])
    #orthogonal pair 3
    line_31 = get_point_or_line(arr[2][0],arr[2][1])
    line_32 = get_point_or_line(arr[2][2],arr[2][3])
    #orthogonal pair 4
    line_41 = get_point_or_line(arr[3][0],arr[3][1])
    line_42 = get_point_or_line(arr[3][2],arr[3][3])
    #orthogonal pair 5
    line_51 = get_point_or_line(arr[4][0],arr[4][1])
    line_52 = get_point_or_line(arr[4][2],arr[4][3])

    return np.array([[line_11,line_12],[line_21,line_22],[line_31, line_32],[line_41,line_42],[line_51,line_52]])

def one_step_system(arr):
    # following the system of equations from the logic
    lines = orthogonal_lines(arr)
    print(lines)
    mat1 = np.empty((0,5),dtype=float)
    mat2 = np.empty((0,0),dtype=float)
    for pair in lines:
        mat1 = np.append(mat1,np.array([[pair[0][0]*pair[1][0],pair[0][0]*pair[1][1]+pair[0][1]*pair[1][0],pair[0][1]*pair[1][1],pair[0][0]+pair[1][0],pair[0][1]+pair[1][1]]]),axis=0)
        mat2 = np.append(mat2,-1)
    return mat1,mat2.reshape((5,1))

def calculate_conic_prime(arr):
    # we get the conic C'_\infty
    mat1,mat2 = one_step_system(arr)
    res = (np.linalg.inv(mat1)@mat2).reshape((5,))
    res /= np.max(res)
    C = [[res[0],res[1],res[3]],[res[1],res[2],res[4]],[res[3],res[4],1]]
    return np.array(C,dtype=float)

def get_H_onestep(C): 
    # get the H matrix from the conic
    V, D, _ = np.linalg.svd(C[:2,:2])
    A_eigenvalues = np.sqrt(np.diag(D))
    A = V.transpose() @ A_eigenvalues @ V
    H = np.zeros((3,3),dtype=float)
    H[:2,:2] = A
    v = np.linalg.inv(A) @ [C[2,0],C[2,1]]
    H[2,:2] = v.transpose()
    H[2,2] = 1
    return np.linalg.inv(H)

def rescale(image, new_width,new_height):
    # this function I use to rescale, in some cases where the image is too big/small/thin
    H = np.array([[new_width/image.shape[1],0,0],[0,new_height/image.shape[0],0],[0,0,1]])
    new_image = apply_homography_resize(image,H)
    return new_image

##### TASK 1 : CORRIDOR

image = cv2.imread("corridor.jpeg")
corridor = np.array([[928, 558],[1306, 488],[1296, 1340],[923, 1131]])
ref = np.array([[928,558],[1328,558],[1328,958],[928,958]])

H = calculate_H_matrix(ref,corridor)
# the only time we use the modification for the apply_homography function
a = apply_homography_resize(image, np.linalg.inv(H),mod=True)

cv2.imwrite("corridor_p2p.jpg",a)

H_v = remove_vanishing_line(corridor)
print(H_v)
b = apply_homography_resize(image, np.linalg.inv(H_v))

new_b = rescale(b, 3000,3000)

cv2.imwrite("corridor_2stepint.jpg",new_b)

H_aff = remove_affine_distortion(corridor)

c = apply_homography_resize(b, H_aff)

cv2.imwrite("corridor_2step.jpg",c)

P = [815,577]
Q = [1073,531]
R = [1069,1214]
S = [811,1069]
# these are the orthogonal lines
o_lines = np.array([[
            # PR- QS, diagonals
            P,R,
            Q,S],
            [
                #PQ - QR
            P,Q,
            Q,R],
            [ # QR - RS
            Q,R,
            R,S],
            [ # RS - SP
            R,S,
            S,P],
            [ # SP - PQ
            S,P,
            P,Q
            
            ]])

conic = calculate_conic_prime(o_lines)
H = get_H_onestep(conic)

d = apply_homography_resize(image, np.linalg.inv(H))
# we rescale here, otherwise it is too thin
new_d = rescale(d, 3000,3000)

cv2.imwrite("corridor_1step.jpg",new_d)

######## TASK 1: BOARD

image2 = cv2.imread("board_1.jpeg")
board = np.array([[71,421],[1223,139],[1354,1954],[423,1799]])
ref = np.array([[70,420],[870,420],[870,1620],[70,1620]])

plt.imshow(image)

H = calculate_H_matrix(ref,board)
a = apply_homography_resize(image2, np.linalg.inv(H))

cv2.imwrite("board_p2p.jpg",a)

H_v = remove_vanishing_line(board)
b = apply_homography_resize(image2, H_v)
cv2.imwrite("board_2stepint.jpg",b)

H_aff = remove_affine_distortion(board)
c = apply_homography_resize(b, H_aff)
cv2.imwrite("board_2step.jpg",c)

P = [811,375]
Q = [916,353]
R = [929,470]
S = [825,490]

o_lines = np.array([[
            # PR- QS, diagonals
            P,R,
            Q,S],
            [
                #PQ - QR
            P,Q,
            Q,R],
            [ # QR - RS
            Q,R,
            R,S],
            [ # RS - SP
            R,S,
            S,P],
            [ # SP - PQ
            S,P,
            P,Q
            
            ]])
conic = calculate_conic_prime(o_lines)
H = get_H_onestep(conic)
d = apply_homography_resize(image2, H)

cv2.imwrite("board_1step.jpg",d)

#### TASK 2: ADOBE BUILDING
adobe_img = cv2.imread("adobe.jpeg")
P = [2104,2816]
Q = [2471,2943]
R = [2475,3437]
S = [2102,3325]
adobe = np.array([P,Q,R,S])
ref = np.array([[2100,2800],[2700,2800],[2700,3400],[2100,3400]])

H = calculate_H_matrix(ref,adobe)
a = apply_homography_resize(adobe_img, np.linalg.inv(H))

cv2.imwrite("adobe_point2point.jpg",a)

H_v = remove_vanishing_line(adobe)
b = apply_homography_resize(adobe_img, H_v)

cv2.imwrite("adobe_2stepint.jpg",b)

H_aff = remove_affine_distortion(adobe)
c = apply_homography_resize(b, H_aff)
# this image is too big in size, so I rescale it here
new_c = rescale(c, 3000,1400)

cv2.imwrite("adobe_2step.jpg",new_c)

o_lines = np.array([[
            # PR- QS, diagonals
            P,R,
            Q,S],
            [
                #PQ - QR
            P,Q,
            Q,R],
            [ # QR - RS
            Q,R,
            R,S],
            [ # RS - SP
            R,S,
            S,P],
            [ # SP - PQ
            S,P,
            P,Q
            
            ]])

conic = calculate_conic_prime(o_lines)
H = get_H_onestep(conic)
d = apply_homography_resize(adobe_img, H)

cv2.imwrite("adobe_onestep.jpg",d)

square_img = cv2.imread("square.jpg")
P = [609,493]
Q = [2182,800]
R = [2086,2364]
S = [652,2665]
square = np.array([P,Q,R,S])
ref = np.array([[200,200],[500,200],[500,500],[200,500]])
H = calculate_H_matrix(ref,square)
a = apply_homography_resize(square_img, np.linalg.inv(H))

cv2.imwrite("square_p2p.jpg",a)

H_v = remove_vanishing_line(square)
b = apply_homography_resize(square_img, H_v)
cv2.imwrite("square_2stepint.jpg",b)

H_aff = remove_affine_distortion(square)
c = apply_homography_resize(b, H_aff)
# this image size is too big, so we resize it here
new_c = cv2.resize(c,(1920,1150))

cv2.imwrite("square_2step.jpg",new_c)

o_lines = np.array([[
            # PR- QS, diagonals
            P,R,
            Q,S],
            [
                #PQ - QR
            P,Q,
            Q,R],
            [ # QR - RS
            Q,R,
            R,S],
            [ # RS - SP
            R,S,
            S,P],
            [ # SP - PQ
            S,P,
            P,Q
            
            ]])
conic = calculate_conic_prime(o_lines)
H = get_H_onestep(conic)
d = apply_homography_resize(square_img, H)
cv2.imwrite("square_onestep.jpg",d)


