import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import open3d.visualization.gui as gui

def main():
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(
        "Add Spheres Example", 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([1, 1, 1, 1])
    scene.scene.scene.set_sun_light(
        [-1, -1, -1],  # direction
        [1, 1, 1],  # color
        100000)
    scene.scene.scene.enable_sun_light(True)
    window.add_child(scene)
    green = rendering.MaterialRecord()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"
    bbox = o3d.geometry.AxisAlignedBoundingBox([-10, -10, -10],
                                               [10, 10, 10])
    scene.setup_camera(60, bbox, [0, 0, 0])
    x = o3d.cpu.pybind.camera.PinholeCameraIntrinsic(2064, 1544, 1109, 1118, 1012, 830)
    extrinsic = scene.scene.camera.get_view_matrix()
    scene.setup_camera(x.intrinsic_matrix, extrinsic, 2064, 1544, bbox)
    scene.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, 1, 0])
    pcd = o3d.io.read_point_cloud('./3d_objects/stanford_bunny.ply')
    pcd.translate([0, 0, 1])
    scene.scene.add_geometry("bunn'", pcd, green)
    gui.Application.instance.run()
    # render.setup_camera(x.intrinsic_matrix, extrinsic, 2064, 1544)
    # render.scene.camera.look_at([0, 0, 0], [0, 0, 10], [0, 1, 0])
    # render = rendering.OffscreenRenderer(2064, 1544)
    #
    # yellow = rendering.MaterialRecord()
    # yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    # yellow.shader = "defaultLit"
    #
    # green = rendering.MaterialRecord()
    # green.base_color = [0.0, 0.5, 0.0, 1.0]
    # green.shader = "defaultLit"
    #
    # grey = rendering.MaterialRecord()
    # grey.base_color = [0.7, 0.7, 0.7, 1.0]
    # grey.shader = "defaultLit"
    #
    # white = rendering.MaterialRecord()
    # white.base_color = [1.0, 1.0, 1.0, 1.0]
    # white.shader = "defaultLit"
    #
    # cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    # cyl.compute_vertex_normals()
    # cyl.translate([-2, 0, 1.5])
    # sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    # sphere.compute_vertex_normals()
    # sphere.translate([-2, 0, 3])
    # pcd = o3d.io.read_point_cloud('./3d_objects/stanford_bunny.ply')
    # pcd.translate([1, -1, 2])
    # box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    # box.compute_vertex_normals()
    # box.translate([-1, -1, 0])
    # solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    # solid.compute_triangle_normals()
    # solid.compute_vertex_normals()
    # solid.translate([0, 0, 1.75])
    #
    # render.scene.add_geometry('bunny', pcd, green)
    # render.scene.add_geometry("cyl", cyl, green)
    # render.scene.add_geometry("sphere", sphere, yellow)
    # render.scene.add_geometry("box", box, grey)
    # render.scene.add_geometry("solid", solid, white)
    # render.setup_camera(60.0, [0, 0, 1], [0, 0, 0], [0, 1, 0])
    # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
    #                                  75000)
    # render.scene.scene.enable_sun_light(True)
    # render.scene.show_axes(True)
    # img = render.render_to_image()
    # print("Saving image at test.png")
    # o3d.io.write_image("test.png", img, 9)
    # extrinsic = render.scene.camera.get_view_matrix()
    # render.setup_camera(x.intrinsic_matrix, extrinsic, 2064, 1544)
    # render.scene.camera.look_at([0, 0, 0], [0, 0, 10], [0, 1, 0])
    # img = render.render_to_image()
    # print("Saving image at test2.png")
    # o3d.io.write_image("test2.png", img, 9)


if __name__ == "__main__":
    main()

