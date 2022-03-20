import numpy as np

def axis_angle(v1, v2):
    v1_n = v1/np.linalg.norm(v1)
    v2_n = v2/np.linalg.norm(v2)

    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1_n, v2_n))

    return axis * angle * 2

def scale_pcd(pcd):
    obox = pcd.get_oriented_bounding_box()
    
    # scale according to longest axis
    # axes = np.array([obox.x_axis, obox.y_axis, obox.z_axis]) # _axis has been removed in version 0.9
                                                               # https://github.com/intel-isl/Open3D/pull/1218
    # axes_lengths = np.linalg.norm(axes, axis=1)
    axes_lengths = obox.extent / 2
    longest_index = np.argmax(axes_lengths)

    # scale
    x_norm_length = 10
    x_scale = x_norm_length / axes_lengths[longest_index]
    obox.scale(x_scale)
    pcd.scale(x_scale)
    
    return (pcd, obox)

def center_scale_pcd(pcd):
    obox = pcd.get_oriented_bounding_box()
    
    # translate so that center is at 0,0,0
    center_translation = np.negative(obox.get_center())
    obox.translate(center_translation)
    pcd.translate(center_translation)
    
    # scale according to longest axis
    # axes = np.array([obox.x_axis, obox.y_axis, obox.z_axis]) # _axis has been removed in version 0.9
                                                               # https://github.com/intel-isl/Open3D/pull/1218
    #axes_lengths = np.linalg.norm(axes, axis=1)
    axes_lengths = obox.extent / 2 # obox.extent => Ausmaße
    longest_index = np.argmax(axes_lengths)

    # scale
    x_norm_length = 50
    x_scale = x_norm_length / axes_lengths[longest_index]
    obox.scale(x_scale)
    pcd.scale(x_scale)
    
    return (pcd, obox)

def normalize_pcd(pcd):
    return center_rotate_scale_pcd(pcd)

def center_rotate_scale_pcd(pcd):
    #obox = pcd.get_oriented_bounding_box()
    aabox = pcd.get_axis_aligned_bounding_box()
    obox = aabox.get_oriented_bounding_box()
    #CreateFromAxisAlignedBoundingBox
    #obox = pcd.get_oriented_bounding_box()
    
    # translate so that center is at 0,0,0
    center_translation = np.negative(obox.get_center())
    obox.translate(center_translation)
    pcd.translate(center_translation)
        
    # align longest axis on x-axis
    # second longest axis on the y-axis
    # shortest axis on the z-axis
    #x_length = np.linalg.norm(obox.x_axis)
    #y_length = np.linalg.norm(obox.y_axis)
    #z_length = np.linalg.norm(obox.z_axis)
    #print(obox.x_axis)
    #axes = np.array([obox.x_axis, obox.y_axis, obox.z_axis]) # _axis has been removed in version 0.9
                                                              # https://github.com/intel-isl/Open3D/pull/1218
    #axes_lengths = np.linalg.norm(axes, axis=1)
    axes_lengths = obox.extent / 2 # obox.extent => Ausmaße
    center = obox.get_center()
    axes = np.array([
        center + [axes_lengths[0], 0, 0],
        center + [0, axes_lengths[1], 0],
        center + [0, 0, axes_lengths[2]]
    ])

    longest_index = np.argmax(axes_lengths)
    shortest_index = np.argmin(axes_lengths)
    second_index = 0
    if longest_index + shortest_index == 2:
        second_index = 1
    elif longest_index + shortest_index == 1:
        second_index = 2
    
    # x (horizontal, y (vertical),z (depth)
    
    #x2 = np.array([-axes_lengths[longest_index], 0, 0])
    #y2 = np.array([0, -axes_lengths[second_index], 0])
    #z2 = np.array([0, 0, -axes_lengths[shortest_index]])
    
    #tm = get_rotation_transformation_matrix(axes[0], axes[1], axes[2], x2, y2, z2)
    #obox.transform(tm)
    #pcd.transform(tm)
    
    #print(obox.x_axis)
    #print(obox.y_axis)
    #print(obox.z_axis)
    
    #return (pcd, obox)
    
    v1_longest = axes[longest_index]
    length_longest = axes_lengths[longest_index]
    v2_longest = np.array([length_longest, 0, 0])
    
    v1_second = axes[second_index]
    length_second = axes_lengths[second_index]
    v2_second = np.array([0, length_second, 0])
    
    v1_latest = axes[shortest_index]
    length_shortest = axes_lengths[shortest_index]
    v2_latest = np.array([0, 0, length_shortest])
    
    x2 = None
    y2 = None
    z2 = None
    if longest_index == 0:
        x2 = v2_longest
    if longest_index == 1:
        y2 = v2_longest
    if longest_index == 2:
        z2 = v2_longest
    if second_index == 0:
        x2 = v2_second
    if second_index == 1:
        y2 = v2_second
    if second_index == 2:
        z2 = v2_second
    if shortest_index == 0:
        x2 = v2_latest
    if shortest_index == 1:
        y2 = v2_latest
    if shortest_index == 2:
        z2 = v2_latest
    
    #print(length_longest)
    
    tm = get_rotation_transformation_matrix(axes[0], axes[1], axes[2], x2, y2, z2)
    # obox.transform(tm) # not supported any more since open3d version 0.9
    pcd.transform(tm)
    #return (pcd, obox)
    
    #axis = axis_angle(v1, v2)
    #obox.rotate(axis, type=geometry.RotationType.AxisAngle)
    #pcd.rotate(axis, type=geometry.RotationType.AxisAngle)

    center_translation = np.negative(obox.get_center())
    obox.translate(center_translation)
    pcd.translate(center_translation)
    
    # scale
    x_norm_length = 70
    x_scale = x_norm_length / axes_lengths[longest_index]
    #print(x_scale)
    obox.scale(x_scale)
    pcd.scale(x_scale, center=False)
    
    # TODO: move center back so that the point cloud surface is aligned at the x-axis
    
    center_translation = np.negative(obox.get_center())
    obox.translate(center_translation)
    pcd.translate(center_translation)
    
    # translate back so that the edge touches the corner?!
    #new_center = np.array([v2_longest[0]/2, v2_second[1]/2, v2_latest[2]/2])
    #obox.translate(new_center)
    #pcd.translate(new_center)
    
    #print(obox.x_axis)
    #print(obox.y_axis)
    return (pcd, obox)

def get_rotation_transformation_matrix(x1, y1, z1, x2, y2, z2):
    M11 = np.dot(x1/np.linalg.norm(x1), x2/np.linalg.norm(x2))
    M12 = np.dot(x1/np.linalg.norm(x1), y2/np.linalg.norm(y2))
    M13 = np.dot(x1/np.linalg.norm(x1), z2/np.linalg.norm(z2))
    M21 = np.dot(y1/np.linalg.norm(y1), x2/np.linalg.norm(x2))
    M22 = np.dot(y1/np.linalg.norm(y1), y2/np.linalg.norm(y2))
    M23 = np.dot(y1/np.linalg.norm(y1), z2/np.linalg.norm(z2))
    M31 = np.dot(z1/np.linalg.norm(z1), x2/np.linalg.norm(x2))
    M32 = np.dot(z1/np.linalg.norm(z1), y2/np.linalg.norm(y2))
    M33 = np.dot(z1/np.linalg.norm(z1), z2/np.linalg.norm(z2))
    
    return np.array([
        [M11, M12, M13, 0],
        [M21, M22, M23, 0],
        [M31, M32, M33, 0],
        [0  , 0  , 0  , 1]
    ])

def get_coordinate_lines():
    points = [[0, 0, 0], [500, 0, 0], [0, 500, 0], [0, 0, 500], [-500, 0, 0], [0, -500, 0], [0, 0, -500]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]]
    colors = [[1, 0, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([line_set])
    return line_set

#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd_original.points).copy())
#pcd.paint_uniform_color([1, 0, 0])
#pcd_original.paint_uniform_color([0, 1, 0])

#line_set = get_coordinate_lines()
#pcd_after, obox = center_rotate_scale_pcd(pcd)
#o3d.visualization.draw_geometries([pcd_original, pcd_after, obox, line_set]) # pcd_original
##o3d.visualization.draw_geometries([pcd_original]) # pcd_original