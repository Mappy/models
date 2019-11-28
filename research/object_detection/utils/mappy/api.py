#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import requests
from io import BytesIO
from PIL import Image

from object_detection.utils.mappy.config import cfg

URL = "http://{}/bo".format(cfg.BO.address_port)
TILES_INDEXES_DICT = ["d", "u", "f", "r", "b", "l", "f"]
TILES_USED_RANGE = range(2, 6)


def next_id():
    try:
        print("{}/marking/next".format(URL))
        response = requests.get("{}/marking/next".format(URL),
                                auth=requests.auth.HTTPBasicAuth(cfg.BO.id, cfg.BO.password))
        next_id = json.loads(response.text.encode("utf-8"))
        return next_id["id"]
    except requests.exceptions.HTTPError as errh:
        print("Http Error:", errh)
        return False
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        return False
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        return False
    except requests.exceptions.RequestException as err:
        print("OOps: Something Else", err)
        return False
    except:
        return False


def get_pano_tiles(im_id, tiles_range=TILES_USED_RANGE):
    tiles = {}
    image_path = construct_path(im_id)
    for i in tiles_range:
        req_url = "{}/images/raw/tile/{}/{}".format(URL, image_path, TILES_INDEXES_DICT[i])

        # tile.show()
        # misc.imsave("/Users/tientranthuong/dev/mappy/boss/ia/models_mappy/tot.jpg", pano)

        # tile = cv2.imdecode(numpy.asarray(bytearray(response.content), dtype=numpy.uint8), cv2.IMREAD_COLOR)
        # im = misc.toimage(pano, channel_axis=2)
        # cv2.imshow("image", misc.fromimage(im))
        # cv2.waitKey()

        tiles[req_url] = i

    return tiles


def get_pano(im_id):
    req_url = "{}/images/fullRaw/tile/{}".format(URL, construct_path(im_id))
    print(req_url)

    response = requests.get(req_url, auth=requests.auth.HTTPBasicAuth(cfg.BO.id, cfg.BO.password))
    # pano = numpy.asarray(bytearray(response.content), dtype=numpy.uint8)
    pano = Image.open(BytesIO(response.content))
    # pano.show()
    return pano


def push_detection(data, im_id):
    req_url = "{}/marking/annotate/{}".format(URL, im_id)
    print(req_url)

    req_header = {'Content-Type': 'application/json'}
    json_data = json.dumps(data)

    response = requests.post(req_url, data=json_data, headers=req_header,
                             auth=requests.auth.HTTPBasicAuth(cfg.BO.id, cfg.BO.password))
    print(response)


def construct_path(id_val):
    id_val = str(id_val)
    path = id_val[:3] + "/" + id_val[3:6] + "/" + id_val[6:9] + "/"
    path += id_val
    return path
