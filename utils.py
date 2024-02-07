import numpy as np

# Defines methods with commonly used functionalities in robotics
class Utils:

    def __init__(self, radians=True):
        self.radians = radians

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
    def inverse_homogeneous_rotation(matrix):
        rotated_matrix = matrix[:3, :3]
        position = matrix[:3, 3]
        inverse_rotation_matrix = Utils.inverse_rotation(rotated_matrix)
        new_position = -1 * Utils.transform_position(position, inverse_rotation_matrix)
        inverted_homogeneous_rotation = np.array([inverse_rotation_matrix, new_position], [0, 0, 0, 1])
        return inverted_homogeneous_rotation

    # Transforms a position by a homogeneous transformation matrix
    @staticmethod
    def transform_position(position, transformation):
        new_position = np.multiply(transformation, np.array([position], [0]))
        return np.array([[new_position[0]], [new_position[1]], [new_position[2]]])

    # Converts a rotation matrix and a translation to a homogeneous matrix
    @staticmethod
    def homogeneous_matrix(rotation_matrix, translation):
        return np.array([[rotation_matrix, translation],[0, 0, 0, 1]])

    # Returns a 2D rotation matrix given an angle
    def rotation_matrix_2d(self, angle):
        if not self.radians:
            angle = self.to_radians(angle)
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Returns a 3D rotation matrix given an angle and a dimension (0: x-axis, 1:y-axis, 2:z-axis))
    def rotation_matrix_3d(self, angle, dimension):
        if not self.radians:
            angle = self.to_radians(angle)
        try:
            if dimension == 0:
                return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            elif dimension == 1:
                return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            elif dimension == 2:
                return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            else:
                raise ValueError("Dimension of a 3D rotation must must be 0 (x-axis), 1 (y-axis) or 2 (z-axis)")
        except Exception as e:
            return e






