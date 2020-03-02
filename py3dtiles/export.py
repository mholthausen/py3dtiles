#! /usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
import getpass
import math
import json
import os
import errno
import numpy as np
from py3dtiles import TriangleSoup, GlTF, B3dm, BatchTable
from .feature_table import FeatureTable
from pyproj import Proj, transform
from pyproj import transform as projtransform

class BoundingBox():
    def __init__(self, minimum, maximum):
        self.min = [float(i) for i in minimum]
        self.max = [float(i) for i in maximum]

    def inside(self, point):
        return ((self.min[0] <= point[0] < self.max[0])
                and (self.min[1] <= point[1] < self.max[1]))

    def center(self):
        return [(i + j) / 2 for (i, j) in zip(self.min, self.max)]

    def add(self, box):
        self.min = [min(i, j) for (i, j) in zip(self.min, box.min)]
        self.max = [max(i, j) for (i, j) in zip(self.max, box.max)]


class Feature():
    def __init__(self, index, box):
        self.index = index
        self.box = box


class Node():
    counter = 0

    def __init__(self, features=[]):
        self.id = Node.counter
        Node.counter += 1
        self.features = features
        self.box = None
        self.children = []

    def add(self, node):
        self.children.append(node)

    def compute_bbox(self):
        print (self)
        self.box = BoundingBox(
            [float("inf"), float("inf"), float("inf")],
            [-float("inf"), -float("inf"), -float("inf")])
        for c in self.children:
            print ('child')
            c.compute_bbox()
            self.box.add(c.box)
        for g in self.features:
            print ('feature')
            self.box.add(BoundingBox(g.box[0], g.box[1]))

    def to_tileset(self, transform):
        self.compute_bbox()
        tiles = {
            "asset": {"version": "1.0"},
            "geometricError": 500,  # TODO
            "root": self.to_tileset_r(500)
        }
        tiles["root"]["transform"] = [round(float(e), 3) for e in transform]
        return tiles

    def to_tileset_r(self, error):

        (c1, c2) = (self.box.min, self.box.max)
        inProj = Proj(init='epsg:4979')
        outProj = Proj(init='epsg:4979')
        xmin,ymin,zmin = c1[0],c1[1],c1[2]
        xmin,ymin,zmin = transform(inProj,outProj,xmin,ymin,zmin)
        xmax,ymax,zmax = c2[0],c2[1],c2[2]
        xmax,ymax,zmax = transform(inProj,outProj,xmax,ymax,zmax)

        xmin,ymin =  xmin / 180 * math.pi, ymin / 180 * math.pi
        xmax,ymax =  xmax  / 180 * math.pi ,ymax / 180 * math.pi

        region = [ymin, xmin, ymax, xmax, c1[2], c2[2]]
        # center = [(c1[i] + c2[i]) / 2 for i in range(0, 3)]
        # xAxis = [(c2[0] - c1[0]) / 2, 0, 0]
        # yAxis = [0, (c2[1] - c1[1]) / 2, 0]
        # zAxis = [0, 0, (c2[2] - c1[2]) / 2]
        # box = [round(x, 3) for x in center + xAxis + yAxis + zAxis]
        # print(box)
        tile = {
            "boundingVolume": {
                "region": region
            },
            "geometricError": error,  # TODO
            "children": [n.to_tileset_r(error / 2.) for n in self.children],
            "refine": "ADD"
        }
        if len(self.features) != 0:
            tile["content"] = {
                "uri": "tiles/{0}.b3dm".format(self.id)
            }

        return tile

    def all_nodes(self):
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes


def tile_extent(extent, size, i, j):
    bbox = BoundingBox(extent[0], extent[1])
    minExtent = [
        bbox.min[0] + i * size,
        bbox.min[1] + j * size]
    maxExtent = [
        bbox.min[0] + (i + 1) * size,
        bbox.min[1] + (j + 1) * size]
    return BoundingBox(minExtent, maxExtent)


# TODO: transform
def arrays2tileset(positions, normals, bboxes, transform, ids=None, doubleSided=False, connection=False):
    print("Creating tileset...")
    maxTileSize = 2000
    indices = [i for i in range(len(positions))]

    # glTF is Y-up, so to get the bounding boxes in the 3D tiles
    # coordinate system, we have to apply a Y-to-Z transform to the
    # glTF bounding boxes
    zUpBboxes = []
    for bbox in bboxes:
        tmp = m = bbox[0]
        M = bbox[1]
        m = [m[0], -m[2], m[1]]
        M = [M[0], -tmp[2], M[1]]
        zUpBboxes.append([m, M])

    # Compute extent
    GxMin = GyMin = float('inf')
    GxMax = GyMax = - float('inf')

    for bbox in bboxes:
        GxMin = min(GxMin, float(bbox[0][0]))
        GyMin = min(GyMin, float(bbox[0][1]))
        GxMax = max(GxMax, float(bbox[1][0]))
        GyMax = max(GyMax, float(bbox[1][1]))

    bboxes4978 = []

    cur = connection.cursor()

    column_name = "geom"
    table_name = "surface"

    for id in ids:
        cur.execute("SELECT ST_3DExtent(ST_Transform({0}, 4978)) FROM {1} WHERE id = {2}".format(column_name, table_name, id))
        extent = cur.fetchall()[0][0]
        extent = [m.split(" ") for m in extent[6:-1].split(",")]
        bboxes4978.append(extent)
        print(bboxes4978)

    # inProj = Proj(init='epsg:4326')
    # outProj = Proj(init='epsg:4978')
    # bboxes4978 = []
    # for bbox in bboxes:
    #     print("----------")
    #     print(bbox)
    #     print("----------")
    #     xMin, yMin = projtransform(inProj, outProj, float(bbox[0][0]), float(bbox[0][1]))
    #     xMax, yMax = projtransform(inProj, outProj, float(bbox[1][0]), float(bbox[1][1]))
    #     bboxes4978.append([[xMin, yMin, bbox[0][2]], [xMax, yMax, bbox[1][2]]])

    extentX = GxMax - GxMin
    extentY = GyMax - GyMin

    # Create quadtree
    tree = Node()
    tilesArr = []

    for idx, box in zip(indices, bboxes):
        tilesArr.append([idx, Feature(idx, box)])

    # Sort geoms by id before creating the b3dm and tileset
    npTiles = np.asarray(tilesArr)
    npTiles = npTiles[np.argsort(npTiles[:, 0])]
    for npTile in npTiles:
        node = Node([npTile[1]])
        tree.add(node)

    # Export b3dm & tileset
    tileset = tree.to_tileset(transform)
    f = open("tileset.json".format(node.id), 'w')
    f.write(json.dumps(tileset))
    print("Creating tiles...")
    nodes = tree.all_nodes()
    identity = np.identity(4).flatten('F')
    try:
        os.makedirs("tiles")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for node in nodes:
        if len(node.features) != 0:
            binarrays = []
            gids = []
            for feature in node.features:
                pos = feature.index
                binarrays.append({
                    'position': positions[pos],
                    'normal': normals[pos][1],
                    'bbox': [[float(i) for i in j] for j in bboxes4978[pos]],
                })
                if ids is not None:
                    gids.append(ids[pos])
            gltf = GlTF.from_binary_arrays(binarrays, identity,
                                           doubleSided=doubleSided,
                                           nMaxMin=normals[pos][0])
            bt = None
            if ids is not None:
                ft = FeatureTable()
                ft.add_property_from_value("BATCH_LENGTH", len(gids))

                bt = BatchTable()
                bt.add_property_from_array("id", gids)

            b3dm = B3dm.from_glTF(gltf, bt=bt, ft=ft).to_array()
            f = open("tiles/{0}.b3dm".format(node.id), 'wb')
            f.write(b3dm)
            f = open("gltf/{0}.glb".format(node.id), 'wb')
            f.write(gltf.to_array())


def divide(extent, geometries, xOffset, yOffset, tileSize,
           featuresPerTile, parent):
    for i in range(0, 2):
        for j in range(0, 2):
            tile = tile_extent(extent, tileSize, i, j)

            geoms = []
            for g in geometries:
                if tile.inside(g.box.center()):
                    geoms.append(g)
            if len(geoms) == 0:
                continue

            if len(geoms) > featuresPerTile:
                node = Node(geoms[0:featuresPerTile])
                parent.add(node)
                divide(tile, geoms[featuresPerTile:len(geoms)],
                       (xOffset + i) * 2, (yOffset + j) * 2,
                       tileSize / 2., featuresPerTile, node)
            else:
                node = Node(geoms)
                parent.add(node)


def wkbs2tileset(wkbs, ids, transform, doubleSided, extent, connection, wkbs4978):
    cur = connection.cursor()

    table_name = 'surface'
    column_name = 'geom'

    extents = []

    for id in ids:
        cur.execute("SELECT ST_3DExtent({0}) FROM {1} WHERE id = {2}".format(column_name, table_name, id))
        extent = cur.fetchall()[0][0]
        extent = [m.split(" ") for m in extent[6:-1].split(",")]
        extents.append(extent)
        print(extent)

    geoms = [TriangleSoup.from_wkb_multipolygon(wkb, extent) for wkb, extent in zip(wkbs4978, extents)]
    positions = [ts.getPositionArray() for ts in geoms]
    normals = [ts.getNormalArray() for ts in geoms]
    # bboxes = [ts.getBbox() for ts in geoms]
    bboxes = [ts.getExtent() for ts in geoms]

    arrays2tileset(positions, normals, bboxes, transform, ids, doubleSided, connection)


def from_db(db_name, table_name, column_name, id_column_name, user_name, host=None, port=None, doubleSided=False):
    user = getpass.getuser() if user_name is None else user_name

    try:
        connection = psycopg2.connect(dbname=db_name, user=user, host=host, port=port)
    except psycopg2.OperationalError:
        pw = getpass.getpass("Postgres password for user {}\n".format(user))
        connection = psycopg2.connect(dbname=db_name, user=user, password=pw, host=host, port=port)

    cur = connection.cursor()

    print("Loading data from database...")
    cur.execute("SELECT ST_3DExtent({0}) FROM {1}".format(column_name, table_name))
    extent = cur.fetchall()[0][0]
    extent = [m.split(" ") for m in extent[6:-1].split(",")]
    offset = [(float(extent[1][0]) + float(extent[0][0])) / 2,
              (float(extent[1][1]) + float(extent[0][1])) / 2,
              (float(extent[1][2]) + float(extent[0][2])) / 2]

    print("-----------EXTENT")
    print(extent)
    print("--------OFFSET")
    print(offset)

    id_statement = ""
    if id_column_name is not None:
        id_statement = "," + id_column_name
    cur.execute("SELECT ST_AsBinary(ST_RotateX(st_transform({0}, 4978), -pi() / 2)), ST_AsBinary(ST_RotateX(ST_Translate({0}, {1}, {2}, {3}), -pi() / 2)),"
                "ST_Area(ST_Force2D({0})) AS weight{5}, id FROM {4} ORDER BY id ASC"
                .format(column_name, -offset[0], -offset[1], -offset[2],
                        table_name, id_statement))
    print("SELECT ST_AsBinary(ST_RotateX(ST_Translate({0}, {1}, {2}, {3}), -pi() / 2)),"
                "ST_Area(ST_Force2D({0})) AS weight{5} FROM {4} ORDER BY id ASC"
                .format(column_name, -offset[0], -offset[1], -offset[2],
                        table_name, id_statement))

    inProj = Proj(init='epsg:4979')
    outProj = Proj(init='epsg:3857')
    xoffset, yoffset = offset[0], offset[1];
    yoffset, xoffset = projtransform(inProj, outProj, xoffset, yoffset)
    inProj = Proj(init='epsg:4979')
    outProj = Proj(init='epsg:4978')
    zoffset = offset[2];
    print ('trans:')
    print (str(xoffset) + '/' + str(yoffset) + '/' + str(zoffset))
    print (projtransform(inProj, outProj, offset[0], offset[1], zoffset))
    unused1, unused2, zoffset = projtransform(inProj, outProj, offset[0], offset[1], zoffset)

    res = cur.fetchall()
    wkbs = [t[1] for t in res]
    wkbs4978 = [t[0] for t in res]
    ids = None
    if id_column_name is not None:
        ids = [t[4] for t in res]
    transform = np.array([
        [1, 0, 0, xoffset],
        [0, 1, 0, yoffset],
        [0, 0, 1, zoffset],
        [0, 0, 0, 1]], dtype=float)
    transform = transform.flatten('F')

    wkbs2tileset(wkbs, ids, transform, doubleSided, extent, connection, wkbs4978)


def from_directory(directory, offset, doubleSided=False):
    # TODO: improvement -> order wkbs by geometry size, similarly to database mode
    offset = (0, 0, 0) if offset is None else offset
    # open all wkbs from directory
    files = os.listdir(directory)
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1] == '.wkb']
    wkbs = []
    for f in files:
        of = open(f, 'rb')
        wkbs.append(of.read())
        of.close()

    transform = np.array([
        [1, 0, 0, offset[0]],
        [0, 1, 0, offset[1]],
        [0, 0, 1, offset[2]],
        [0, 0, 0, 1]], dtype=float)
    transform = transform.flatten('F')
    wkbs2tileset(wkbs, None, transform, doubleSided)


def init_parser(subparser, str2bool):
    descr = 'Generate a tileset from a set of geometries'
    parser = subparser.add_parser('export', help=descr)

    group = parser.add_mutually_exclusive_group()

    d_help = 'name of the directory containing the geometries'
    group.add_argument('-d', metavar='DIRECTORY', type=str, help=d_help)

    o_help = 'offset of the geometries (only with -d)'
    parser.add_argument('-o', nargs=3, metavar=('X', 'Y', 'Z'), type=float, help=o_help)

    D_help = 'database name'
    group.add_argument('-D', metavar='DATABASE', type=str, help=D_help)

    t_help = 'table name'
    parser.add_argument('-t', metavar='TABLE', type=str, help=t_help)

    c_help = 'geometry column name'
    parser.add_argument('-c', metavar='COLUMN', type=str, help=c_help)

    parser.add_argument('-i', metavar='IDCOLUMN', type=str, help=c_help)

    u_help = 'database user name'
    parser.add_argument('-u', metavar='USER', type=str, help=u_help)

    H_help = 'database host'
    parser.add_argument('-H', metavar='HOST', type=str, help=H_help)

    P_help = 'database port'
    parser.add_argument('-P', metavar='PORT', type=int, help=P_help)

    ds_help = 'glTF doubleSided'
    parser.add_argument('-ds', metavar='DOUBLESIDED', type=bool, help=ds_help)


def main(args):
    if args.D is not None:
        if args.t is None or args.c is None:
            print('Error: please define a table (-t) and column (-c)')
            exit()

        from_db(args.D, args.t, args.c, args.i, args.u, args.H, args.P, args.ds)
    elif args.d is not None:
        from_directory(args.d, args.o)
    else:
        raise NameError('Error: database or directory must be set')
