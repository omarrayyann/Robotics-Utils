import numpy as np
import scipy

# Defines methods with commonly used functionalities in robotics
class Utils:

    # Converts an angle in degree to radians
    @staticmethod
    def to_radians(angle):
        return angle*np.pi/180

    # Converts an angle in radians to degrees
    @staticmethod
    def to_degrees(angle):
        return angle*180/np.pi

    # Computes the trace of a given matrix
    @staticmethod
    def trace(matrix):
        trace = 0.0
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                trace = trace + matrix[r][c]
        return trace

    # Checks if a given matrix is a rotation matrix
    @staticmethod
    def is_rotation_matrix(matrix):
        is_orthogonal = np.allclose(np.transpose(matrix).dot(matrix), np.identity(matrix.shape[0]))
        is_determinant_one = np.isclose(abs(np.linalg.det(matrix)), 1)
        return is_orthogonal and is_determinant_one

    # Returns the inverse of a rotation matrix (it's transpose)
    @staticmethod
    def inverse_rotation(matrix):
        inverted_rotation = np.transpose(matrix)
        return inverted_rotation

    # Invert a homogeneous transformation matrix
    @ staticmethod
    def inverse_homogeneous_matrix(matrix):
        rotated_matrix = matrix[:3, :3]
        position = matrix[:3, 3].reshape(-1, 1)
        inverse_rotation_matrix = Utils.inverse_rotation(rotated_matrix)
        new_position = -1 * Utils.apply_rotation(inverse_rotation_matrix, position)
        inverted_homogeneous_matrix = np.block([[inverse_rotation_matrix, new_position], [0, 0, 0, 1]])
        return inverted_homogeneous_matrix

    # Transforms a position by a homogeneous transformation matrix
    @staticmethod
    def transform_position(transformation,position):
        return (transformation @ np.block([[position], [1]]))[0:3]

    # Transforms a vector by a homogeneous transformation matrix
    @staticmethod
    def transform_vector(transformation, vector):
        return (transformation @ np.block([[vector], [0]]))[0:3]

    # Converts a rotation matrix and a translation to a homogeneous matrix (both rotation and translation happens based on old axis)
    @staticmethod
    def homogeneous_matrix(rotation_matrix, translation):
        if rotation_matrix is None:
            rotation_matrix = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        if translation is None:
            translation = np.array([[0.0],[0.0],[0.0]])
        return np.block([
            [rotation_matrix, translation],
            [np.zeros((1, 3)), 1]
        ])

    # Applies rotation on a position
    @staticmethod
    def apply_rotation (rotation, position):
        return rotation @ position

    # Returns a 2D rotation matrix given an angle
    @staticmethod
    def rotation_matrix_2d(angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Returns a 3D rotation matrix in the x dimension given an angle
    @staticmethod
    def rotation_matrix_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    # Returns a 3D rotation matrix in the y dimension given an angle
    @staticmethod
    def rotation_matrix_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    # Returns a 3D rotation matrix in the z dimension given an angle
    @staticmethod
    def rotation_matrix_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    # Returns a 3D rotation matrix in the z dimension given an angle
    @staticmethod
    def translation(x,y,z):
        return np.array([[x],[y],[z]])


    # Returns a rotation matrix using the euler angles (alpha, beta, gamma)
    @staticmethod
    def rotation_matrix_rpy(roll, pitch, yaw):
        rotation_matrix_x = Utils.rotation_matrix_x(roll)
        rotation_matrix_y = Utils.rotation_matrix_y(pitch)
        rotation_matrix_z = Utils.rotation_matrix_z(yaw)
        euler_rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
        return euler_rotation_matrix

    # Returns Roll (x, gamma), Pitch (y, beta) and Yaw (z, aplha) (ZYX Euler Rotations) given a rotation matrix
    @staticmethod
    def find_rpy(rotation_matrix):
        pitch = np.arctan2(-rotation_matrix[2][0], (rotation_matrix[0][0]*rotation_matrix[0][0] + rotation_matrix[1][0]*rotation_matrix[1][0])**0.5)
        yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])
        roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])
        return roll, pitch, yaw

    # Finds skew symmetric matrix given a 3D vector [x, y, z]
    @staticmethod
    def to_skew_3d(vector):
        return np.array([
            [0.0, -vector[2][0], vector[1][0]],
            [vector[2][0], 0.0, -vector[0][0]],
            [-vector[1][0], vector[0][0], 0.0]])
    
    # Finds skew symmetric matrix given a 6D vector [x, y, z]
    @staticmethod
    def to_skew_6d(vector):
        top = vector[0:3,:]
        bottom = vector[3:6,:]
        return np.block([
            [Utils.to_skew_3d(top), bottom],
            [0.0,0.0,0.0,0.0]
        ])

    # Finds the 3D vector [x, y, z] from a skew symmetric matrix
    @staticmethod
    def from_skew(matrix):
        return np.array([matrix[2][1], matrix[0][2], -matrix[0][1]])

    # Checks if a given matrix is a skew symmetric matrix
    @staticmethod
    def is_skew_symmetric(matrix):
        if matrix[0][1] != -matrix[1][0]:
            return False
        if matrix[0][2] != -matrix[2][0]:
            return False
        if matrix[1][2] != -matrix[2][2]:
            return False
        return True

    # Returns the ross product two 3x3 matrices
    @staticmethod
    def cross_product_3d(vector1, vector2):
        return Utils.vector_to_ssm(vector1) @ vector2

    # Finds spatial angular velocity and returns [w]
    @staticmethod
    def find_spatial_angular_velocity(rot, d_rot):
        return d_rot @ rot.T
    
    # Finds body angular velocity and returns [w]
    @staticmethod
    def find_body_angular_velocity(rot, d_rot):
        return rot.T @ d_rot
    
    # Finds linear velocity given an angular velocity and position in a simular frame
    @staticmethod
    def find_linear_velocity(angular_velocity, position):
        return Utils.cross_product_3d(angular_velocity, position)


    # Returns a twist given an angular velocity and linear velocity
    @staticmethod
    def to_twist_vector(linear_velocity, angular_velocity):
        return np.block([[angular_velocity],[linear_velocity]])
    
    # Returns a twist given an angular velocity and linear velocity
    @staticmethod
    def to_twist_matrix(linear_velocity, angular_velocity):
        twist_vector = Utils.to_twist_vector(linear_velocity, angular_velocity)
        return Utils.twist_vector_to_twist_matrix(twist_vector)
    

    # Returns a homogenus transformation given a twist [[w0],[w1],[w2],[v0],[v1],v[2]]
    @staticmethod
    def twist_vector_to_twist_matrix(vector):
        return Utils.exp(Utils.to_skew_6d(vector))
    
    # Exponential of a matrix
    def exp(matrix):
        return scipy.linalg.expm(matrix)