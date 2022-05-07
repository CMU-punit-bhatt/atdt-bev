
CARLA_CLASSES_I_TO_NAME = {
    0: 'Unlabeled',
    1: 'Building',
    2: 'Fence',
    3: 'Other',
    4: 'Pedestrian',
    5: 'Pole',
    6: 'RoadLine',
    7: 'Road',
    8: 'SideWalk',
    9: 'Vegetation',
    10: 'Vehicles',
    11: 'Wall',
    12: 'TrafficSign',
    13: 'Sky',
    14: 'Ground',
    15: 'Bridge',
    16: 'RailTrack',
    17: 'GuardRail',
    18: 'TrafficLight',
    19: 'Static',
    20: 'Dynamic',
    21: 'Water',
    22: 'Terrain'
}

CARLA_CLASSES_NAME_TO_RGB = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (100, 40, 40),
    3: (55, 90, 80),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45, 60, 150),
    22: (145, 170, 100)
}


NUSCENES_CLASSES_NAME_TO_I = {
    'None': 0,
    'animal': 1,
    'human.pedestrian.adult': 2,
    'human.pedestrian.child': 3,
    'human.pedestrian.construction_worker': 4,
    'human.pedestrian.personal_mobility': 5,
    'human.pedestrian.police_officer': 6,
    'human.pedestrian.stroller': 7,
    'human.pedestrian.wheelchair': 8,
    'movable_object.barrier': 9,
    'movable_object.debris': 10,
    'movable_object.pushable_pullable': 11,
    'movable_object.trafficcone': 12,
    'static_object.bicycle_rack': 13,
    'vehicle.bicycle': 14,
    'vehicle.bus.bendy': 15,
    'vehicle.bus.rigid': 16,
    'vehicle.car': 17,
    'vehicle.construction': 18,
    'vehicle.emergency.ambulance': 19,
    'vehicle.emergency.police': 20,
    'vehicle.motorcycle': 21,
    'vehicle.trailer': 22,
    'vehicle.truck': 23,
    'flat.driveable_surface': 24,
    'vehicle.ego': 31
}

NUSCENES_I_TO_CLASS_NAME = {
    0: 'None',
    1: 'animal',
    2: 'human.pedestrian.adult',
    3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker',
    5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer',
    7: 'human.pedestrian.stroller',
    8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier',
    10: 'movable_object.debris',
    11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone',
    13: 'static_object.bicycle_rack',
    14: 'vehicle.bicycle',
    15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid',
    17: 'vehicle.car',
    18: 'vehicle.construction',
    19: 'vehicle.emergency.ambulance',
    20: 'vehicle.emergency.police',
    21: 'vehicle.motorcycle',
    22: 'vehicle.trailer',
    23: 'vehicle.truck',
    24: 'flat.driveable_surface',
    31: 'vehicle.ego'
}

NUSCENES_CARLA_MAP = {
    0: 0,
    1: 20,
    2: 4,
    3: 4,
    4: 4,
    5: 3,
    6: 4,
    7: 3,
    8: 3,
    9: 2,
    10: 3,
    11: 3,
    12: 3,
    13: 10,
    14: 10,
    15: 10,
    16: 10,
    17: 10,
    18: 10,
    19: 10,
    20: 10,
    21: 10,
    22: 10,
    23: 10,
    24: 7,
    31: 10,
}