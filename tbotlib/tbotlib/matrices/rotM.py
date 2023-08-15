import numpy as np

def rotM(theta_x, theta_y = None, theta_z = None, order='xyz'):

    '''
    Calculates the rotation matrix R from the angles theta_x, theta_y, theta_z
    theta_x: rotation around the x-axis in deg
    theta_y: rotation around the y-axis in deg
    theta_z: rotation around the z-axis in deg
    order:   order of the rotation 

    https://en.wikipedia.org/wiki/Euler_angles
    '''

    if theta_z is None and theta_y is None:

        theta_x = np.deg2rad(theta_x)

        R = np.array([[np.cos(theta_x), -np.sin(theta_x)],
                      [np.sin(theta_x),  np.cos(theta_x)]])
    
    else:
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)

        Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]]) 
        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]]) 
        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        
        if order == 'xyz':
            R = Rz @ Ry @ Rx
        elif order == 'yxz':
            R = Rz @ Rx @ Ry
        
    return R


def rotM2(a, b):

    '''
    Calculates rotation matrix to rotate vector a to vector b
    a: vector
    b: vector
    R: rotation matrix
    '''

    #get unit vectors
    u_a = a/np.sqrt(np.sum(np.square(a)))
    u_b = b/np.sqrt(np.sum(np.square(b)))

    #get products
    cos_t = np.sum(u_a * u_b)
    sin_t = np.sqrt(np.sum(np.square(np.cross(u_a,u_b)))) #magnitude

    #get new unit vectors
    u = u_a
    v = u_b - np.sum(u_a * u_b)*u_a
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(u_a, u_b)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C

    return R


def decompose(R, order='xyz'):

    '''
    Calculates the angles theta_x, theta_y, theta_z from the rotation matrix R
    theta_x: rotation around the x-axis in deg
    theta_y: rotation around the y-axis in deg
    theta_z: rotation around the z-axis in deg
    order:   order of the rotation
    '''

    if order == 'yxz':
        #decomposes as RzRxRy <- rotation by y first
        theta_z = np.arctan2(-R[0,1], R[1,1])                         
        theta_y = np.arctan2(-R[2,0], R[2,2])                         
        theta_x = np.arctan2( R[2,1], np.sqrt(R[2,0]**2 + R[2,2]**2)) 

    elif order == 'xyz':
        #decomposes as RzRyRx <- rotation by x first
        theta_x = np.arctan2( R[2,1], R[2,2])                                                
        theta_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+ R[2,2]**2))
        theta_z = np.arctan2( R[1,0], R[0,0])                           

    theta_x = np.rad2deg(theta_x)
    theta_y = np.rad2deg(theta_y)
    theta_z = np.rad2deg(theta_z)

    return theta_x, theta_y, theta_z

if __name__ == '__main__':
    R = rotM(10,20,30)
    print(R)
    print(decompose(R))

    R = rotM(10,20,30,order='yxz')
    print(R)
    print(decompose(R,order='yxz'))