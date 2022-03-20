import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import time
import os


class PointDropExperiment:

    def __init__(
            self,
            module,
            classifier,
            grad_cam,
            testdataloader,
            test_dataset,
            num_drops=1024,
            show_visualization=False,
            file_prefix="",
            steps=250,
            steps_heatmap_update=False,  # defaults to steps
            num_iterations=10,
            update_cam=True,
            alternative_colors=False,
            show_marker=True,
            create_png=False,
            random_drop=True,
            high_drop=True,
            low_drop=True,
            print_progress=False,
            use_cam_points=False,
            results_dir='./',
            device='cpu',
            title_prefix='',
            is_mesh=False,
            remove_edges=False,
            drop_in_batches=False):
        self.drop_in_batches = drop_in_batches
        self.module = module
        self.title_prefix = title_prefix
        self.results_dir = results_dir
        self.device = device
        self.steps = steps
        if steps_heatmap_update is False:
            self.steps_heatmap_update = steps
        else:
            self.steps_heatmap_update = steps_heatmap_update
        self.classifier = classifier
        self.grad_cam = grad_cam
        self.testdataloader = testdataloader
        self.show_marker = show_marker

        self.show_visualization = show_visualization
        self.num_drops = num_drops
        self.num_classified = 0
        self.num_iterations = num_iterations
        self.alternative_colors = alternative_colors

        #num_data_points = num_drops + 1
        num_data_points = 1 # for 0 drop values
        for i in range(1, self.num_drops + 1):
            if self.measure_condition(i):
                num_data_points += 1

        self.random_correct = np.zeros(num_data_points)
        self.random_target_confidence = np.zeros(num_data_points)
        self.random_loss = np.zeros(num_data_points)

        self.high_correct = np.zeros(num_data_points)
        self.high_target_confidence = np.zeros(num_data_points)
        self.high_loss = np.zeros(num_data_points)

        self.low_correct = np.zeros(num_data_points)
        self.low_target_confidence = np.zeros(num_data_points)
        self.low_loss = np.zeros(num_data_points)

        self.file_prefix = file_prefix
        self.test_dataset = test_dataset

        # create point cloud and reuse it
        self.pcd = o3d.geometry.PointCloud()
        self.update_cam = update_cam

        self.create_png = create_png

        self.random_drop = random_drop
        self.high_drop = high_drop
        self.low_drop = low_drop

        self.print_progress = print_progress

        self.use_cam_points = use_cam_points

        self.iteration_index = 0

        self.foo = 1
        self.is_mesh = is_mesh
        self.remove_edges = remove_edges

    def run_experiment(self):
        self.classifier.eval()
        self.module.reset_seeds()

        start_i = self.load_cached_experiment() + 1
        print("start_i", start_i)

        for i, data in enumerate(self.testdataloader):
            if i < start_i:
                continue
            self.iteration_index = i
            start = time.time()
            input, target = self.module.get_data(data)
            targex_idx = target.view(-1)[0].cpu().item()

            self.drop_experiment(input, targex_idx)

            dur = time.time() - start
            if self.print_progress:
                print('experiment:', self.num_iterations, 'pcds', self.num_drops, 'drops', self.steps, 'steps')
                print(i, 'done in', dur, 'seconds')
                print('estimated time untill complete:', (self.num_iterations - i) * dur, 'seconds')

            self.cache_result(i)

        self.save_results()
        self.plot_results()

    def cache_result(self, i):
        np.savez(os.path.join(self.results_dir, self.file_prefix + 'exp_cache.npz'),
                 random_target_confidence=self.random_target_confidence,
                 high_target_confidence=self.high_target_confidence,
                 low_target_confidence=self.low_target_confidence,
                 random_loss=self.random_loss,
                 high_loss=self.high_loss,
                 low_loss=self.low_loss,
                 random_correct=self.random_correct,
                 high_correct=self.high_correct,
                 low_correct=self.low_correct,
                 i=i)

    def load_cached_experiment(self):
        if not os.path.exists(os.path.join(self.results_dir, self.file_prefix + 'exp_cache.npz')):
            return -1
        cache = np.load(os.path.join(self.results_dir, self.file_prefix + 'exp_cache.npz'))
        self.random_target_confidence = cache['random_target_confidence']
        self.high_target_confidence = cache['high_target_confidence']
        self.low_target_confidence = cache['low_target_confidence']
        self.random_loss = cache['random_loss']
        self.high_loss = cache['high_loss']
        self.low_loss = cache['low_loss']
        self.random_correct = cache['random_correct']
        self.high_correct = cache['high_correct']
        print("cache['high_correct']", cache['high_correct'])
        self.low_correct = cache['low_correct']
        print("cache['low_correct']", cache['low_correct'])
        self.num_classified = cache['i'] + 1
        return cache['i']

    def drop_point(self, input, idx):
        if self.is_mesh:
            self.vertice_to_edge = {}  # vertice => [edge_id]
            for e_id in range(len(input.edge_index[0])):
                for vert in [input.edge_index[0, e_id], input.edge_index[1, e_id]]:
                    vert = vert.item()
                    if vert not in self.vertice_to_edge:
                        self.vertice_to_edge[vert] = []
                    if e_id not in self.vertice_to_edge[vert]:
                        self.vertice_to_edge[vert].append(e_id)

            input.x[idx, :] = 0
            input.pos[idx, :] = 0
            if self.remove_edges:
                edge_mask = np.ones(input.edge_attr.shape[0], dtype=np.bool)
                if idx in self.vertice_to_edge:
                    for e_id in self.vertice_to_edge[idx]:
                        edge_mask[e_id] = False
                else:
                    print("vertice with no edges!!!")
                input.edge_attr = input.edge_attr[edge_mask]
                input.edge_index = input.edge_index.T[edge_mask].T
        else:
            input[:, :, idx] = 0  # shift point to center

    def drop_points(self, input, idx_list):
        if self.is_mesh:
            self.vertice_to_edge = {}  # vertice => [edge_id]
            for e_id in range(len(input.edge_index[0])):
                for vert in [input.edge_index[0, e_id], input.edge_index[1, e_id]]:
                    vert = vert.item()
                    if vert not in self.vertice_to_edge:
                        self.vertice_to_edge[vert] = []
                    if e_id not in self.vertice_to_edge[vert]:
                        self.vertice_to_edge[vert].append(e_id)

            for idx in idx_list:
                input.x[idx, :] = 0
                input.pos[idx, :] = 0
            if self.remove_edges:
                edge_mask = np.ones(input.edge_attr.shape[0], dtype=np.bool)
                for idx in idx_list:
                    if idx in self.vertice_to_edge:
                        for e_id in self.vertice_to_edge[idx]:
                            edge_mask[e_id] = False
                    else:
                        print("vertice with no edges!!!")
                input.edge_attr = input.edge_attr[edge_mask]
                input.edge_index = input.edge_index.T[edge_mask].T
        else:
            for idx in idx_list:
                input[:, idx, :] = 0  # shift point to center

    def drop_experiment(self, input, target_idx):
        if self.is_mesh:
            N = input.pos.size(0)
        else:
            N = input.size(1)

        target = torch.tensor([target_idx], dtype=torch.long).to(self.device)

        self.grad_cam.set_iteration_index(self.iteration_index)
        self.classifier.zero_grad()
        cam, _, _ = self.grad_cam(input, target_idx)
        if self.use_cam_points:  # p++ ppd experiment
            input = self.grad_cam.points

        initial_result = self.grad_cam.classifier_output
        initial_class_id = initial_result.data.argmax(1)[0]

        # add 0 drop confidence values
        initial_confidence = self.get_confidence(initial_result, target_idx)
        self.random_target_confidence[0] += initial_confidence
        self.high_target_confidence[0] += initial_confidence
        self.low_target_confidence[0] += initial_confidence

        # add 0 drop loss values
        initial_loss = self.get_loss(initial_result[0], target)
        self.random_loss[0] += initial_loss
        self.high_loss[0] += initial_loss
        self.low_loss[0] += initial_loss

        # add 0 drop accuracy values
        initial_correct = 1 if initial_class_id == target_idx else 0
        self.random_correct[0] += initial_correct
        self.high_correct[0] += initial_correct
        self.low_correct[0] += initial_correct

        if self.random_drop:
            # copy input
            points = input.clone()
            if not self.is_mesh:
                points = points.detach()

            # get self.num_drops random integer numbers between 0 and points.size(2)
            rand_idx = np.random.choice(N, size=self.num_drops, replace=False)
            j = 1

            drop_idx = []
            for i in range(self.num_drops):
                idx = rand_idx[i]
                if self.drop_in_batches:
                    drop_idx.append(idx)
                    if self.measure_condition(i + 1):
                        self.drop_points(points, drop_idx)
                        drop_idx = []
                else:
                    self.drop_point(points, idx)
                    # points[:, :, idx] = 0 # shift point to center

                if self.measure_condition(i + 1):
                    # measure metrics
                    self.classifier.zero_grad()
                    result = self.classifier(points)
                    if not self.is_mesh:
                        result = result[0]
                    #result = self.grad_cam.only_classify(points)
                    class_id = result.data.argmax(1)[0]
                    self.random_target_confidence[j] += self.get_confidence(result, target_idx)
                    self.random_loss[j] += self.get_loss(result, target)
                    self.random_correct[j] += 1 if class_id == target_idx else 0
                    j += 1
                    # del result

            if not self.is_mesh:
                if self.num_drops < points.shape[2]:
                    self.show_pcd(points)
            del points

        if self.high_drop:
            # copy input
            points = input.clone()
            if not self.is_mesh:
                points = points.detach()
            hcam = np.copy(cam)
            # hcam_gate is necessary to ignore the cam values of the already ignored points
            hcam_add_gate = np.zeros(hcam.size, dtype=np.dtype(np.float32))
            k = 1
            drop_idx = []

            for i in range(self.num_drops):
                idx = np.argmax(hcam)
                # set cam value to zero so that the next argmax call the return the next lower value
                hcam[idx] = -1.5
                hcam_add_gate[idx] = -2.5
                if self.drop_in_batches:
                    drop_idx.append(idx)
                    if self.measure_condition(i + 1):
                        self.drop_points(points, drop_idx)
                        drop_idx = []
                else:
                    self.drop_point(points, idx)
                # points[:, :, idx] = 0 # shift point to center
                result = None

                if self.heatmap_update_condition(i + 1):
                    # update cam
                    hcam, _, _ = self.grad_cam(points, target_idx)
                    if np.isnan(hcam).any():
                        hcam = np.zeros(hcam.size)
                    self.update_heatmap(points, hcam)
                    hcam = hcam + hcam_add_gate
                    result = self.grad_cam.classifier_output

                if self.measure_condition(i + 1):
                    # measure metrics
                    if result is None:
                        self.classifier.zero_grad()
                        result = self.classifier(points)
                        if not self.is_mesh:
                            result = result[0]
                    class_id = result.data.argmax(1)[0]
                    self.high_target_confidence[k] += self.get_confidence(result, target_idx)
                    self.high_loss[k] += self.get_loss(result, target)
                    self.high_correct[k] += 1 if class_id == target_idx else 0
                    k = k + 1

            if not self.is_mesh:
                if self.num_drops < points.shape[2]:
                    self.show_pcd(points)

        if self.low_drop:
            # copy input
            points = input.clone()
            if not self.is_mesh:
                points = points.detach()
            lcam = np.copy(cam)
            # lcam_gate is necessary to ignore the cam values of the already ignored points
            lcam_add_gate = np.zeros(lcam.size)
            j = 1
            drop_idx = []

            for i in range(self.num_drops):
                idx = np.argmin(lcam)
                # set cam value to one so that the next argmin call the return the next higher value
                lcam[idx] = 1.5
                lcam_add_gate[idx] = 1.5
                if self.drop_in_batches:
                    drop_idx.append(idx)
                    if self.measure_condition(i + 1):
                        self.drop_points(points, drop_idx)
                        drop_idx = []
                else:
                    self.drop_point(points, idx)
                # points[:, :, idx] = 0 # shift point to center
                result = None

                if self.heatmap_update_condition(i + 1):
                    # update cam
                    lcam, _, _ = self.grad_cam(points, target_idx)
                    if np.isnan(lcam).any():
                        lcam = np.zeros(lcam.size)
                    self.update_heatmap(points, lcam)
                    lcam = lcam + lcam_add_gate
                    result = self.grad_cam.classifier_output

                if self.measure_condition(i + 1):
                    # measure metrics
                    if result is None:
                        self.classifier.zero_grad()
                        result = self.classifier(points)
                        if not self.is_mesh:
                            result = result[0]
                    class_id = result.data.argmax(1)[0]
                    self.low_target_confidence[j] += self.get_confidence(result, target_idx)
                    self.low_loss[j] += self.get_loss(result, target)
                    self.low_correct[j] += 1 if class_id == target_idx else 0
                    j += 1

                if i == 250:
                    self.show_pcd(points)

                if i == 500:
                    self.show_pcd(points)

                if i == 1000:
                    self.show_pcd(points)

            if not self.is_mesh:
                if self.num_drops < points.shape[2]:
                    self.show_pcd(points)

        self.num_classified += 1

    def measure_condition(self, i):
        # i is the index of the currently dropped point in the range of [1, self.num_points]
        return i % self.steps == 0 or i == self.num_drops

    def heatmap_update_condition(self, i):
        return self.update_cam and i % self.steps_heatmap_update == 0

    def save_results(self):
        np.savez(self.get_file_name(),
                 num_drops=np.array([self.num_drops]),
                 num_classified=np.array([self.num_classified]),
                 random_correct=self.random_correct,
                 random_target_confidence=self.random_target_confidence,
                 random_loss=self.random_loss,
                 high_correct=self.high_correct,
                 high_target_confidence=self.high_target_confidence,
                 high_loss=self.high_loss,
                 low_correct=self.low_correct,
                 low_target_confidence=self.low_target_confidence,
                 low_loss=self.low_loss)


    def get_file_name(self):
        suffix = '-hm_updated' if self.update_cam else ''
        return os.path.join(self.results_dir, f"mat-{self.file_prefix}{self.num_iterations}pcds-{self.num_drops}drops-{self.steps}steps{suffix}.npz")


    def load_results(self):
        mat = np.load(self.get_file_name())
        self.num_drops = mat['num_drops'][0]
        self.num_classified = mat['num_classified'][0]
        self.random_correct = mat['random_correct']
        self.random_target_confidence = mat['random_target_confidence']
        self.random_loss = mat['random_loss']
        self.high_correct = mat['high_correct']
        self.high_target_confidence = mat['high_target_confidence']
        self.high_loss = mat['high_loss']
        self.low_correct = mat['low_correct']
        self.low_target_confidence = mat['low_target_confidence']
        self.low_loss = mat['low_loss']

    def load_by_npz_name(self, file_path):
        # file_path = f"./tmp/mat-{file_name}.npz"
        mat = np.load(file_path)
        print("file_path", file_path)
        self.num_drops = mat['num_drops'][0]
        self.num_classified = mat['num_classified'][0]
        self.random_correct = mat['random_correct']
        self.random_target_confidence = mat['random_target_confidence']
        self.random_loss = mat['random_loss']
        self.high_correct = mat['high_correct']
        self.high_target_confidence = mat['high_target_confidence']
        self.high_loss = mat['high_loss']
        self.low_correct = mat['low_correct']
        self.low_target_confidence = mat['low_target_confidence']
        self.low_loss = mat['low_loss']

    def plot_results(self):
        measures = [
            ('Accuracy', self.random_correct, self.high_correct, self.low_correct),
            ('Confidence', self.random_target_confidence, self.high_target_confidence, self.low_target_confidence),
            ('Loss', self.random_loss, self.high_loss, self.low_loss),
        ]

        colors = ['g', 'r', 'b']
        if self.alternative_colors:
            colors = ['g', 'orange', '#9122D1']

        # create x axis
        x_values = [0]
        for i in range(self.num_drops):
            if self.measure_condition(i + 1):
                x_values.append(i + 1)
            # else ignore as no data for this index has been collected
        #x = np.arange(self.num_drops + 1)  # for 0 drops + 1
        x = np.array(x_values)

        for m in measures:
            plt.figure()
            if self.show_marker is True:
                plt.plot(x, m[1] / self.num_classified, color=colors[0], label='rand-drop', ls='-', marker='o')
                plt.plot(x, m[2] / self.num_classified, color=colors[1], label='high-drop', ls='-', marker='+')
                plt.plot(x, m[3] / self.num_classified, color=colors[2], label='low-drop', ls='-', marker='v')
            else:
                plt.plot(x, m[1] / self.num_classified, color=colors[0], label='rand-drop', ls='-')
                plt.plot(x, m[2] / self.num_classified, color=colors[1], label='high-drop', ls='-')
                plt.plot(x, m[3] / self.num_classified, color=colors[2], label='low-drop', ls='-')

            if m[0] != 'Loss':
                plt.plot(x, np.ones(len(x_values)) * (1/16), color='c', label='random guess', ls='--')
            plt.title(f"{self.title_prefix}{self.num_classified} pcds, {self.num_drops} drops, {self.steps} step, {'hm updated' if self.update_cam else 'hm not updated'}")
            if m[0] == 'Loss':
                plt.ylim(0.0)
            else:
                plt.ylim(0.0, 1.05)
            plt.xlim(0.0)
            plt.xlabel('Number of Points Dropped')
            plt.ylabel(m[0])
            #plt.xticks(np.arange(0, self.num_drops, 100))
            #if m[0] != 'loss':
            #    plt.yticks(np.arange(0, 1., 0.1))
            plt.grid()
            plt.legend()
            suffix = '-hm_updated' if self.update_cam else ''
            plt.savefig(f"{self.results_dir}/{self.file_prefix}{self.num_classified}pcds-{self.num_drops}drops-{self.steps}steps-{m[0]}{suffix}.png")

    def get_area_under_the_curve(self):
        """
        This method calculates the area under the curces for Accuracy, Confidence and Loss for the
        three dropping methods random, low dropping and high dropping.

        The returned 3x3 matrix contains in the rows the measured values for Accuracy, Confidence and Loss in this
        order and in the rows the random, low and high dropping results.
        """
        measures = [
            ('Accuracy', self.random_correct, self.low_correct, self.high_correct),
            ('Confidence', self.random_target_confidence, self.low_target_confidence, self.high_target_confidence),
            ('Loss', self.random_loss, self.low_loss, self.high_loss),
        ]

        # create x axis
        x_values = [0]
        for i in range(self.num_drops):
            if self.measure_condition(i + 1):
                x_values.append(i + 1)
        x_values = np.array(x_values)

        area_matrix = np.zeros((3, 3))

        for i in range(len(measures)):
            for j in range(1, len(measures[i])):
                y_values = measures[i][j] / self.num_classified
                total_area = 0.0

                for s in range(0, x_values.shape[0] - 1):
                    # if s == 39:
                    #     print("asd")
                    # calculate are between x[i] and x[i + 1]
                    x = x_values[s]
                    x_next = x_values[s + 1]
                    v = y_values[s]
                    v_next = y_values[s + 1]
                    v_min = min([v, v_next])
                    step_size = (x_next - x)
                    square = step_size * v_min
                    triangle = (abs(v_next - v) * step_size) / 2
                    total_area += square + triangle

                # print("i", i, "j", j)
                area_matrix[i][j - 1] = total_area
                # print("total_are", metric, total_area)

        return area_matrix

    def get_loss(self, input, target):
        #return np.square(input.cpu().data.numpy() - target.cpu().data.numpy()).sum()
        return F.cross_entropy(input.view(1, -1), target)


    def get_confidence(self, output, target_idx):
        return torch.nn.functional.softmax(output).cpu()[0, target_idx]


    def update_heatmap(self, input, cam_mask):
        if self.is_mesh:
            return
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = np.squeeze(heatmap, axis=1)
        heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]  # only one line Change

        #print("input ---", input)
        self.pcd.colors = o3d.utility.Vector3dVector(
            [heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]] for j in range(input.size(1)))


    def show_pcd(self, draw_points):
        if not self.show_visualization: return
        #print(draw_points)
        self.pcd.points = o3d.utility.Vector3dVector(draw_points[0].transpose(0, 1).cpu().data.numpy())

        if self.create_png:
            self.custom_draw_geometry(self.pcd, point_size=5.0)
        else:
            o3d.visualization.draw_geometries([self.pcd])

    def custom_draw_geometry(self, pcd, x=-190, y=140, point_size=7.0, show_coordinate_frame=False,
                             output_path='./pc-images'):
        # saving does not seem to work on all systems, alternatively save the point cloud including the colors
        # and move them to a system which can create pngs
        create_images = False  # else save the point cloud as ply

        if create_images:
            # Visualize Point Cloud
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)

            opt = vis.get_render_option()  # needs to be called after create_window has been called
            opt.point_size = point_size  # default is 5.0
            opt.show_coordinate_frame = show_coordinate_frame
            opt.background_color = np.asarray([1, 1, 1])

            ctr = vis.get_view_control()
            # ctr.change_field_of_view(step=90)
            ctr.rotate(x, y)
            ctr.scale(-4)

            # Updates
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture image
            time.sleep(1)
            time_suffix = int(round(time.time() * 1000))
            # vis.capture_screen_image(os.path.join(output_path, f"pc-{time_suffix}.png"), do_render=True)

            vis.run()
            # Close
            vis.destroy_window()
        else:
            # alternatively save the point cloud
            # self.results_dir was output_path before
            o3d.io.write_point_cloud(os.path.join(self.results_dir, f"pc-{self.file_prefix}-{self.iteration_index}-{self.foo}.ply"), pcd)
            self.foo = self.foo + 1
