import numpy as np

"""
These are fuctions for processing data in the format of structured mesh.
"""
def concat_zones(zones) -> np.ndarray:
    upper = np.flip(np.concatenate([
        zones[14][:, :-1],
        zones[16][:, :-1],
        zones[18][:, :-1],
        zones[20][:, :-1],
        zones[22],
    ], axis=1), axis=1)

    lower = np.flip(np.concatenate([
        np.flip(zones[0].transpose(1, 0, 2, 3),1) [:, :-1],
        zones[1][:, :-1],
        zones[4][:, :-1],
        zones[6][:, :-1],
        zones[8][:, :-1],
        zones[10],
    ], axis=1), axis=0)

    inner = np.concatenate([
        upper[:, :-1], 
        lower
    ], axis=1)

    lower_far = np.flip(np.concatenate([
        zones[2][:, :-1],
        zones[5][:, :-1],
        zones[7][:, :-1],
        zones[9][:, :-1],
        zones[12],
    ], axis=1), axis=0)

    upper_far = np.flip(np.concatenate([
        np.flip(zones[3].transpose(1, 0, 2, 3), axis=0)[:, :-1],
        zones[13][:, :-1],
        zones[15][:, :-1],
        zones[17][:, :-1],
        zones[19][:, :-1],
        zones[21],
    ], axis=1), axis=1)

    outer = np.concatenate([
        upper_far[:, :-1], 
        lower_far
    ], axis=1)

    Z = np.concatenate([
        inner, outer
    ])    

    return Z

def convert_to_np(data_dir: str, output_dir: str):
    """
    Convert the .dat files to numpy ndarrays. Output is of shape (H, W, I, C).
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith('.dat'):
            path = os.path.join(data_dir, filename)
            with open(path, 'r') as f:
                zones = []
                for line in f:
                    if line.startswith('ZONE'):
                        settings = {}
                        while True:
                            line = f.readline().strip()
                            if line.startswith("DT"): break
                            for pair in line.replace('\n', '').split(','):
                                k, v = pair.split('=')
                                settings[k.strip()] = v.strip()
                        I, J, K = int(settings['I']), int(settings['J']), int(settings['K'])
                        zone = []
                        for _ in range(I * J * K):
                            zone.append([float(x) for x in f.readline().strip().split(' ')])
                        zone = np.array(zone)
                        zones.append(zone.reshape(K, J, I, zone.shape[-1]))
                Z = concat_zones(zones)
                aoa = float(settings.get('AUXDATA Common.AngleOfAttack').strip('"'))
                mach = float(settings.get('AUXDATA Common.ReferenceMachNumber').strip('"'))
                target = os.path.join(output_dir, filename.replace('deg.dat',f'_{mach}'))
                np.save(target, Z)

def process_zones(filename: str):
    with open(filename, 'r') as f:
        zones = []
        for line in f:
            if line.startswith('ZONE'):
                settings = {}
                while True:
                    line = f.readline().strip()
                    if line.startswith("DT"): break
                    for pair in line.replace('\n', '').split(','):
                        k, v = pair.split('=')
                        settings[k.strip()] = v.strip()
                I, J, K = int(settings['I']), int(settings['J']), int(settings['K'])
                zone = []
                for _ in range(I * J * K):
                    zone.append([float(x) for x in f.readline().strip().split(' ')])
                zone = np.array(zone)
                zones.append(zone.reshape(K, J, I, zone.shape[-1]))
        Z = concat_zones(zones)
        aoa = float(settings.get('AUXDATA Common.AngleOfAttack').strip('"'))
        mach = float(settings.get('AUXDATA Common.ReferenceMachNumber').strip('"'))
        np.save(filename.replace('deg.dat',f'_{mach}'), Z)
    return Z, zones

def generate_mesh(contour, save_path, visualize=False):
    """
    This function generates mesh for an airfoil from specified options.
    TODO: add more options and finer control.
    """
    import gmsh
    gmsh.initialize()
    gmsh.model.add('new model')
    lc = 0.002
    
    airfoil_points = []
    for point in contour:
        x, y = point
        airfoil_points.append(gmsh.model.geo.add_point(x, y, 0, lc))
    airfoil_points.append(airfoil_points[0])
    
    top = gmsh.model.geo.add_point(0, 10, 0, 500*lc)
    center = gmsh.model.geo.add_point(0, 0, 0, 500*lc)
    bottom = gmsh.model.geo.add_point(0, -10, 0, 500*lc)
    arc = gmsh.model.geo.add_circle_arc(top, center, bottom)

    top_right = gmsh.model.geo.add_point(10, 10, 0, 1000*lc)
    bottom_right = gmsh.model.geo.add_point(10, -10, 0, 1000*lc)
    rec = gmsh.model.geo.add_polyline([top, top_right, bottom_right, bottom])
    
    airfoil = gmsh.model.geo.add_spline(airfoil_points)
    
    surface = gmsh.model.geo.add_plane_surface([
        gmsh.model.geo.add_curve_loop([arc, -rec]), # farfield
        gmsh.model.geo.add_curve_loop([airfoil]) 
    ])
    gmsh.model.geo.synchronize()

    airfoil_tag = gmsh.model.add_physical_group(1, [airfoil])
    gmsh.model.set_physical_name(1, airfoil_tag, 'airfoil')
    farfield_tag = gmsh.model.add_physical_group(1, [arc, -rec])
    gmsh.model.set_physical_name(1, farfield_tag, 'farfield')
    surface_tag = gmsh.model.add_physical_group(2, [surface])
    gmsh.model.set_physical_name(2, surface_tag, 'surface')

    gmsh.model.mesh.generate(2)
    gmsh.write(save_path)

    if visualize:
        gmsh.fltk.run()
    gmsh.finalize()

def read_dat(file_name):
    profile = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                x, y = line.strip().split()
                profile.append((float(x), float(y)))
    return profile

