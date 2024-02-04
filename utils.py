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





