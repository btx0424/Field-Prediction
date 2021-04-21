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