import open3d as o3d
import numpy as np
from collections import defaultdict
from concave_hull import concave_hull, concave_hull_indexes
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt



def get_poisson_mesh_from_point_cloud(point_cloud):
    """
    Generates a mesh from a point cloud using the Poisson Surface Reconstruction algorithm.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.

    Returns:
        open3d.geometry.TriangleMesh: The generated mesh.
    """
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    return mesh

def clean_mesh(mesh, obtain_main_component=True):
    """
    Cleans a mesh by removing degenerate triangles, duplicates, non-manifold edges and outliers.

    Args:
        mesh (open3d.geometry.TriangleMesh): The input mesh

    Returns:
        open3d.geometry.TriangleMesh: The cleaned mesh
    """
    mesh_pcd = mesh.sample_points_poisson_disk(5000)
    # Create mesh from the point cloud
    new_mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(mesh_pcd, depth=9)
    # Clean the mesh
    new_mesh.compute_triangle_normals()
    new_mesh.compute_vertex_normals()
    # Remove degenerate and duplicate triangles
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    # Remove duplicate vertices
    new_mesh.remove_duplicated_vertices()
    # Remove non-manifold edges
    new_mesh.remove_non_manifold_edges()
    # Remove outliers
    new_mesh = handle_mesh_outliers(new_mesh, obtain_main_component)

    return new_mesh

def handle_mesh_outliers(mesh,  obtain_main_component):
    """
    Handle mesh outliers by removing connected components that are not the largest one.

    Parameters:
        mesh (open3d.geometry.TriangleMesh): The input mesh.
        obtain_main_component (bool): If True, remove all connected components except the largest one.
                                       If False, return the removed connected components for visualization.

    Returns:
        open3d.geometry.TriangleMesh: The mesh with outliers removed.

    This function clusters the triangles in the input mesh into connected components and identifies the largest connected component.
    It then removes all the triangles that belong to connected components other than the largest one, unless obtain_main_component is False,
    in which case it returns the removed connected components for visualization.

    Note: The input mesh is modified in-place.

    Example usage:
        mesh = handle_mesh_outliers(mesh, obtain_main_component=True)
    """
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # Find the largest connected component
    largest_cluster_idx = cluster_n_triangles.argmax()
    if obtain_main_component:
        triangles_to_remove = triangle_clusters != largest_cluster_idx
    else:
        # Get removed components for visualization
        triangles_to_remove = triangle_clusters == largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    return mesh


def create_percentage_based_bounding_box(original_bbox, percentage_range, longest_axis_index, R):
    """
    Creates a new bounding box with the same orientation and axis lengths as the original bounding box,
    but with positions adjusted along the longest axis based on the specified percentage range.
    
    Parameters:
    - original_bbox: Open3D OrientedBoundingBox object representing the original bounding box
    - percentage_range: Tuple containing the percentage range (e.g., (0.2, 0.8) for 20% to 80% range)
    - longest_axis_index: Index of the longest axis
    - R: Rotational matrix of the original bounding box
    
    Returns:
    - new_bbox: Open3D OrientedBoundingBox object representing the new bounding box
    """
    # Get the lengths along the principal axes
    lengths = original_bbox.extent
    
    # Calculate the start and end positions along the longest axis based on the percentage range
    start_percentage, end_percentage = percentage_range
    start_position = lengths[longest_axis_index] * start_percentage
    end_position = lengths[longest_axis_index] * end_percentage
    
    # Create a new bounding box with the same orientation and axis lengths
    new_bbox = o3d.geometry.OrientedBoundingBox(center=original_bbox.center, R=R, extent=lengths)
    
    # Adjust positions along the longest axis
    new_extent = np.array(lengths)
    new_extent[longest_axis_index] = end_position - start_position
    new_bbox.extent = new_extent
    
    return new_bbox

def move_box_along_longest_axis(bounding_box, distance, axis_index):
    """
    Moves the center of the bounding box by a specified distance along the longest axis.
    
    Parameters:
    - bounding_box: Open3D OrientedBoundingBox object representing the bounding box
    - distance: Distance to move the center along the longest axis
    
    Returns:
    - translated_bounding_box: Open3D OrientedBoundingBox object with the translated center
    """
    # Determine the longest axis of the bounding box
    
    # Calculate the translation vector along the longest axis
    translation_vector = np.zeros(3)
    translation_vector[axis_index] = distance 
    
    # Apply rotation to the translation vector
    rotated_translation_vector = np.dot(translation_vector, bounding_box.R.T)
    
    # Translate the center of the bounding box along the longest axis
    translated_center = bounding_box.get_center() + rotated_translation_vector
    
    # Create a new bounding box with the translated center
    translated_bounding_box = o3d.geometry.OrientedBoundingBox(translated_center, bounding_box.R, bounding_box.extent)
    
    return translated_bounding_box


def get_volume_of_cave_section(original_bounding_box, mesh, segment_size, segment_digits, longest_axis_index):
    """
    Computes the volume of a cave section by cropping a mesh and computing convex hulls and volumes of each partition.

    Parameters:
    - original_bounding_box (open3d.geometry.OrientedBoundingBox): The bounding box of the original mesh
    - mesh (open3d.geometry.TriangleMesh): The input mesh
    - segment_size (float): The size of each segment for the cave section
    - segment_digits (int): The number of digits to round the position to
    - longest_axis_index (int): The index of the longest axis (0, 1, or 2)

    Returns:
    - volume (float): The total volume of the cave section
    - boxes (list): The list of bounding boxes used for cropping the mesh
    - meshes (list): The list of cropped meshes
    - hulls (list): The list of convex hulls computed for each partition
    """
    # For tracking the position of the box, 0.5 is the start, 0.0 is middle and -0.5 is end of the longest axis
    position = 0.5
    boxes = []
    meshes = []
    hulls = []
    volume = 0
    # Create a bounding box with the same orientation and axis lengths as the original bounding box,
    # but with a smaller size along the longest axis
    segment_of_bounding_box = create_percentage_based_bounding_box(original_bounding_box, (0.0, segment_size), longest_axis_index, original_bounding_box.R)
    while position >= -0.5:
        # Move the bounding box along the longest axis by a specified offset
        translated_bounding_box = translate_bounding_box(original_bounding_box, longest_axis_index, position, segment_of_bounding_box)
        # For visualization
        # translated_bounding_box.color = [0,1,0]
        boxes.append(translated_bounding_box)
        # Crop the mesh based on the box
        cave_partition = mesh.crop(translated_bounding_box)
        meshes.append(cave_partition)
        # Compute the convex hull for getting the volume
        convex_hull = cave_partition.compute_convex_hull()
       #o3d.visualization.draw_geometries([translated_bounding_box,mesh,convex_hull[0]])
        try:
            volume += convex_hull[0].get_volume()
            hulls.append(convex_hull[0])
        except:
            pass
        # Move to the next segment
        position = round(position - segment_size, segment_digits)
    return volume, boxes, meshes, hulls

def translate_bounding_box(original_bounding_box, longest_axis_index, position, segment_of_bounding_box):
    move_offset = (original_bounding_box.extent[longest_axis_index] - segment_of_bounding_box.extent[longest_axis_index]) * position
    translated_bounding_box = move_box_along_longest_axis(segment_of_bounding_box, move_offset, longest_axis_index)
    return translated_bounding_box

def compute_surface_area(mesh):
    """
    Computes the surface area of a mesh.

    Parameters:
    - mesh (open3d.geometry.TriangleMesh): The input mesh.

    Returns:
    - area (float): The total surface area of the mesh.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    area = 0.0
    for triangle in triangles:
        v0, v1, v2 = vertices[triangle]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    
    return area

def create_path_skeleton_line(point_cloud, longest_axis_index):
    """
    Creates a path skeleton line from a given point cloud and the index of the longest axis.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        longest_axis_index (int): The index of the longest axis.

    Returns:
        open3d.geometry.LineSet: The path skeleton line.
    """
    bounding_box = point_cloud.get_oriented_bounding_box()
    small_box = create_percentage_based_bounding_box(bounding_box, (0.0, 0.1), longest_axis_index, bounding_box.R)
    clouds = []
    box_position = 0.5
    while box_position >= -0.5:
            # Move the bounding box along the longest axis by a specified offset
            translated_bounding_box = translate_bounding_box(bounding_box, longest_axis_index, box_position, small_box)
            # For visualization
            # translated_bounding_box.color = [0,1,0]
            # Crop the mesh based on the box
            cave_partition = point_cloud.crop(translated_bounding_box)
            clouds.append(cave_partition)
            box_position = round(box_position - 0.1, 1)

    # Obtain geometric centers of small clouds, then connect them
    geometric_centers = []
    for cloud in clouds:
        points = np.asarray(cloud.points)
        center = np.mean(points, axis=0)
        geometric_centers.append(center)

    lines = []
    for i in range(len(geometric_centers) - 1):
        lines.append([i,i+1])

    # Create Open3D line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(geometric_centers)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def create_cave_skeleton(point_cloud, mesh, longest_axis_index):
    """
    Creates a cave skeleton from a point cloud and a mesh.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        mesh (open3d.geometry.TriangleMesh): The input mesh.
        longest_axis_index (int): The index of the longest axis of the mesh.

    Returns:
        tuple: A tuple containing two elements:
            - line_skeleton (open3d.geometry.LineSet): The line skeleton of the cave.
            - mesh_skeleton (open3d.geometry.LineSet): The mesh skeleton of the cave.
    """
    line_skeleton = create_path_skeleton_line(point_cloud, longest_axis_index)
    # Filter the mesh so it has less triangles
    mesh = mesh.filter_smooth_simple(number_of_iterations=5) 
    mesh.compute_vertex_normals()   
    # Create mesh skeleton
    mesh_skeleton = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    # Make it gray
    mesh_skeleton.paint_uniform_color([0.841,0.841, 0.841])
    return line_skeleton, mesh_skeleton

def compute_total_line_length(lineset):
    """
    Compute the total length of lines in a line set.

    Parameters:
    - lineset: The line set containing points and lines.
    
    Returns:
    - total_length: The sum of the lengths of all lines in the line set.
    """
    # Get the points and lines from the lineset
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)
    
    # Calculate the length of each line
    line_lengths = np.linalg.norm(points[lines[:, 0]] - points[lines[:, 1]], axis=1)
    
    # Sum up the lengths of all lines
    total_length = np.sum(line_lengths)
    
    return total_length


def group_points_by_x_coordinate(mesh):
    """
    Group points in the mesh based on distinct x coordinates.

    Parameters:
    - mesh: Open3D TriangleMesh object

    Returns:
    - grouped_point_clouds: List of Open3D PointCloud objects, where each object contains points with the same x coordinate.
    """
    # Convert mesh vertices to numpy array
    vertices = np.asarray(mesh.vertices)

    # Group points based on their x coordinates
    groups = defaultdict(list)
    for vertex in vertices:
        x_coord = vertex[0]
        groups[x_coord].append(vertex)

    grouped_point_clouds = []
    for i, (x_coord, points) in enumerate(groups.items()):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        grouped_point_clouds.append(cloud)
    return grouped_point_clouds

def perimeter(points):
    """
    Calculate the perimeter of a polygon defined by the given points.

    Parameters:
    - points (numpy.ndarray): An array of shape (N, 2) representing the coordinates of the points.

    Returns:
    - perimeter_length (float): The length of the perimeter of the polygon.
    """
    perimeter_length = 0
    for i in range(len(points) - 1):
        perimeter_length += np.linalg.norm(points[i] - points[i + 1])
    perimeter_length += np.linalg.norm(points[0] - points[len(points) - 1])
    return perimeter_length

def area(points):
    """
    Calculate the area of a polygon defined by the given points using the Shoelace formula.

    Parameters:
    - points (numpy.ndarray): An array of shape (N, 2) representing the coordinates of the points.

    Returns:
    - area (float): The area of the polygon.
    """
    num_points = len(points)
    area = 0
    for i in range(num_points):
        j = (i + 1) % num_points
        area += points[i, 0] * points[j, 1] - points[j, 0] * points[i, 1]
    return abs(area) / 2

def obtain_roundness(grouped_point_clouds):
    """
    Calculates the circularities of convex hulls and concave hulls in a list of grouped point clouds.

    Parameters:
        grouped_point_clouds (list): A list of Open3D PointCloud objects representing grouped point clouds.

    Returns:
        tuple: A tuple containing two lists. The first list contains the circularities of convex hulls, and the second list contains the circularities of concave hulls.
    """
    circularities_of_convex_hulls = []
    circularities_of_concave_hulls = []

    for cloud in grouped_point_clouds:
        point_cloud_np_without_x = np.asarray(cloud.points)[:, 1:]
        concave_h = concave_hull(point_cloud_np_without_x)
        convex_h = ConvexHull(point_cloud_np_without_x)
        sorted_convex_hull_points = point_cloud_np_without_x[convex_h.vertices]
        ch_perim = perimeter(sorted_convex_hull_points)
        ch_area = area(sorted_convex_hull_points)
        circularities_of_convex_hulls.append(4 * np.pi * ch_area / (ch_perim * ch_perim))
        ch_perim = perimeter(concave_h)
        ch_area = area(concave_h)
        circularities_of_concave_hulls.append(4 * np.pi * ch_area / (ch_perim * ch_perim))

            

    return circularities_of_convex_hulls, circularities_of_concave_hulls

def crop_main_cloud(main_point_cloud, downsampled_point_cloud, segment_size, longest_axis_index, segment_position):
    """
    Crops a main point cloud based on a specified segment size, longest axis index, and segment position.

    Parameters:
    - main_point_cloud (open3d.geometry.PointCloud): The main point cloud to be cropped.
    - downsampled_point_cloud (open3d.geometry.PointCloud): The downsampled point cloud used to calculate the bounding box.
    - segment_size (float): The size of the segment.
    - longest_axis_index (int): The index of the longest axis.
    - segment_position (float): The position of the segment.

    Returns:
    - cropped_cloud (open3d.geometry.PointCloud): The cropped point cloud.
    """
    # position 0.5 is leftmost
    bounding_box = downsampled_point_cloud.get_minimal_oriented_bounding_box()
    small_box = create_percentage_based_bounding_box(bounding_box, (0.0, segment_size), longest_axis_index, bounding_box.R)
    translated_bounding_box = translate_bounding_box(bounding_box, longest_axis_index, segment_position, small_box)
    cropped_cloud = main_point_cloud.crop(translated_bounding_box)
    return cropped_cloud