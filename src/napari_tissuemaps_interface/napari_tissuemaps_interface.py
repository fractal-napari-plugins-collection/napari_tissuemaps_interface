"""
Main module containing the reader for multi-scale (pyramidal) TIFF files,
as well as the napari widget interface
"""
import os
from xml.dom import minidom
import json
from typing import List
from io import BytesIO
import dask
from napari_plugin_engine import napari_hook_implementation
import numpy as np
import requests
import PIL
from napari.types import LayerDataTuple
from qtpy.QtWidgets import QLineEdit  # pylint: disable=E0611
from magicgui.widgets import FunctionGui

from .lazy_array import LazyArray  # pylint: disable=E0401

TILE_SIZE = 256  # hard-coded in TissueMAPs


def authenticate(url, username, password):
    """
    Helper function that returns an authentication token

    :param url: The url of the authentication service
    :param username: The username to be used for authentication
    :param password: The password to be used for authentication
    :return: The access token
    """
    response = requests.post(
        url + '/auth',
        data=json.dumps({'username': username, 'password': password}),
        headers={'content-type': 'application/json'},
    )
    response.raise_for_status()
    data = response.json()
    return data['access_token']


def http_get(url, api_uri, token, **params):
    """
    Helper function to perform an http get, with optional parameters

    :param url: The url of the endpoint
    :param api_url: The URI of the specific api
    :param token: The access token
    :param params: kwargs for optional parameter to the http get
    :return: The full requests response
    """
    response = requests.get(
        url + '/' + api_uri, params=params,
        headers={'Authorization': 'JWT ' + token},
    )
    response.raise_for_status()
    return response


def get_data(url, api_uri, token, **params):
    """
    Helper function to perform an http get to a json enpoint which contains a 'data' field
    Supports optinal parameters for get with optional parameters

    :param url: The url of the endpoint
    :param api_url: The URI of the specific api
    :param token: The access token
    :param params: kwargs for optional parameter to the http get
    :return: The content of the 'data' field in the json response
    """
    response = http_get(url, api_uri, token, **params)
    data = response.json()
    return data['data']


def tissuemaps_interface(url, token, experiment_id, channel_layer_id):
    # pylint: disable=R0914
    """
    Function which reads an image layer from the TissueMAPs API.
    Given NAPARI_OCTREE==1, it returns a multi-scale (pyramidal) image as a delayed Dask
    array. Otherwise, it returns a high-resolution image as numpy array.
    Note: The later will download the full image into memory!

    :param url: The base url of the TissueMAPs server
    :param token: The authentication token obtained by the TissueMAPs server
    :param experiment_id: The TissueMAPs experiment id
    :param channel_layer_id: The TissueMAPs channel layer id

    :return: A tiled multi-scale image if NAPARI_OCTREE==1,
             otherwise a numpy array with the high-resolution image
    """
    class LazyTiledTMArray(LazyArray):
        """
        A numpy-like array which lazily loads tiles from a TissueMAPS server.
        """
        # pylint: disable=R0903
        def __init__(self, shape, dtype, tile_size, zoom):
            super().__init__(shape, dtype, tile_size)
            self.zoom = zoom

        @dask.delayed
        def read_tile(self, y_tile, x_tile):
            '''
            Reads a tile from a TissuMAPS server

            :param y_tile: the y coordinate of the tile
            :param x_tile: the x coordinate of the tile

            :return: numpy array with the the cooresponding tile updated
            '''
            tiles_resp = http_get(
                url, 'api/experiments/' + str(experiment_id) + '/channel_layers/' +
                str(channel_layer_id) + '/tiles', token, x=x_tile, y=y_tile, z=self.zoom
            )
            img = PIL.Image.open(BytesIO(tiles_resp.content))
            data = np.zeros((self.tile_size, self.tile_size))
            data[:img.size[1], :img.size[0]] = np.asarray(img)
            return data

    channel_layers = get_data(url, 'api/experiments/' +
                              str(experiment_id) + '/channel_layers',
                              token)

    channel_layer = next(item for item in channel_layers if item["id"] ==
                         channel_layer_id)

    image_data = {}
    image_data['image_height'] = channel_layer['image_size']['height']
    image_data['image_width'] = channel_layer['image_size']['width']
    image_data['max_zoom'] = channel_layer['max_zoom']

    pyramid = []
    for zoom in reversed(range(1, image_data['max_zoom'] + 1)):
        if image_data['image_width'] < TILE_SIZE:
            break

        array = LazyTiledTMArray(
            shape=(image_data['image_height'], image_data['image_width']),
            dtype=np.uint8,
            tile_size=TILE_SIZE,
            zoom=zoom
        )

        if 'NAPARI_OCTREE' not in os.environ or os.environ['NAPARI_OCTREE'] != '1':
            # given we don't have a spatial index (e.g. an octree), we can
            # directly return the heighest resolution.
            # NOTE: this will download the full image into memory!
            array = np.asarray(array)
            return array

        pyramid.append(array)
        image_data['image_height'] //= 2
        image_data['image_width'] //= 2

    return pyramid


def tissuemaps_connector(path):
    """
    Function which reads an XML specifying TissueMAPS credential and experiment/channel_layer data
    and return a multi-scale (pyramidal) JPEG from TissueMAPs api as delayed Dask
    array.

    :param path: The path of the image
    :return: List of LayerData tuple
    """

    xmldoc = minidom.parse(path)
    auth_data = {}
    auth_data['url'] = xmldoc.getElementsByTagName('url')[0].attributes['url'].value
    auth_data['user'] = \
        xmldoc.getElementsByTagName('user')[0].attributes['name'].value
    auth_data['password'] = \
        xmldoc.getElementsByTagName('user')[0].attributes['password'].value
    auth_data['token'] = \
        authenticate(auth_data['url'], auth_data['user'], auth_data['password'])

    query_data = {}
    query_data['experiment_id'] = \
        xmldoc.getElementsByTagName('layerdata')[0].attributes['experiment_id'].value

    query_data['channel_layer_id'] = \
        xmldoc.getElementsByTagName('layerdata')[0].attributes['channel_layer_id'].value

    pyramid = tissuemaps_interface(auth_data['url'], auth_data['token'],
                                   query_data['experiment_id'],  query_data['channel_layer_id'])

    # kwargs = {}
    return [(pyramid, {})]


# Widget code

class TissueMAPSGetTokenWidget(FunctionGui):
    # pylint: disable=R0901
    # Disabled check for number of ancestors, since this class will have 17 ancestors,
    # way more than the 5 suggested by pylint
    """
    Inner widget to handle connection to a given TissueMAPS server.
    This widget stores username, password and access token
    """
    def __init__(self, value=None, name="tm_connector", **kwargs):
        # pylint: disable=W0613
        if value is None:
            value = ("", "", "")
        url, username, password = value
        super().__init__(
            TissueMAPSGetTokenWidget.apply,
            call_button=False,
            layout='vertical',
            param_options={
                "url": {"widget_type": "LineEdit"},
                "username": {"widget_type": "LineEdit"},
                "password": {"widget_type": "LineEdit"},
                "add_button": {
                    "widget_type": "PushButton", "text": "Connect",
                }
            },
            name=name
        )

        self.password.native.setEchoMode(QLineEdit.Password)
        self.url.value = url
        self.username.value = username
        self.password.value = password
        self.token = ""

        self.native.layout().setContentsMargins(0, 0, 0, 0)

        @self.add_button.changed.connect
        def on_press_import_button(event):
            # pylint: disable=W0613
            self.token = authenticate(self.url.value, self.username.value, self.password.value)

    def __setitem__(self, key, value):
        """Prevent assignment by index."""
        raise NotImplementedError("magicgui.Container does not support item setting.")

    @staticmethod
    def apply(url="", username="", password="", add_button=True):
        # pylint: disable=W0613
        """
        Dummy function to respect the FunctionGui logic. Not used since
        call_button is False in this widget
        """

    @property
    def value(self):
        """
        Associates the value field of the TissueMAPSGetTokenWidget to the access token
        """
        return self.token


class TissueMAPSConnectionWidget(FunctionGui):
    # pylint: disable=R0901,R0903
    # Disabled check for number of ancestors, since this class will have 17 ancestors,
    # way more than the 5 suggested by pylint
    """
    Main widget to manage TissueMAPS data. It contains the TissueMAPSGetTokenWidget
    """
    def __init__(self, value=None, **kwargs):
        # pylint: disable=W0613
        super().__init__(
            self.apply,
            call_button="Load Data",
            layout="vertical",
            param_options={
                "token": {"widget_type": TissueMAPSGetTokenWidget, "name": "tm_connector"},
                "experiment_name": {"choices": [""]},
                "channel_layer": {"choices": [""]},
                "load_all_channels": {"enabled": True, "text": "Load all channels"},
            },
        )

        def get_experiments(*args):
            # pylint: disable=W0613
            if len(self.experiments_data) > 0:
                return [item['name'] for item in self.experiments_data]

            return [""]

        def search_id(name):
            """
            Function that creates a dictionary of experiments names and thair ids.

            :param token: experiment name
            :return: the correspondent id
            """
            dict_exp = [{'name': item['name'], 'id': item['id']} for item in self.experiments_data]
            id_exp = next(item for item in dict_exp if item["name"] == name)['id']
            return id_exp

        def get_channel_layers(*args):
            # pylint: disable=W0613
            if len(self.channel_layer_data) > 0:
                return self.channel_layer_data

            return[""]

        @self.tm_connector.changed.connect
        def update_experiments(event):
            # pylint: disable=W0613
            if self.tm_connector.token != "":
                resp = get_data(self.tm_connector.url.value, "/api/experiments",
                                self.tm_connector.token)
                self.experiments_data = resp
                self.experiment_name.reset_choices()

        @self.experiment_name.changed.connect
        def update_channel_layer_id(event):
            # pylint: disable=W0613
            id_exp = search_id(self.experiment_name.value)
            resp = get_data(self.tm_connector.url.value, 'api/experiments/' +
                            str(id_exp) + '/channel_layers',
                            self.tm_connector.token)
            self.channel_layer_data = resp
            self.channel_layer.reset_choices()

        self.experiments_data = []
        self.channel_layer_data = []
        self.experiment_name._default_choices = get_experiments
        self.channel_layer._default_choices = get_channel_layers

        self.native.layout().addStretch()

    def search_id(self, name):
        """
        Function that creates a dictionary of experiments names and thair ids.

        :param token: experiment name
        :return: the correspondent id
        """
        dict_exp = [{'name': item['name'], 'id': item['id']} for item in self.experiments_data]
        id_exp = next(item for item in dict_exp if item["name"] == name)['id']
        return id_exp

    def __setitem__(self, key, value):
        """Prevent assignment by index."""
        raise NotImplementedError("magicgui.Container does not support item setting.")
        # pylint: disable=C0301

    def apply(self, token=("", "", ""),
              experiment_name="",
              load_all_channels=True,
              channel_layer="") -> List[LayerDataTuple]:
        # pylint: disable=W0613
        """
        Function executed when the "Load Data" button is pressed.
        It calls the tissuemaps tissuemaps_interface and returns a napari
        Image layer

        :param token: The access token for querying TissueMAPS
        :param experiment_name: The name of a TissueMAPS experiment
        :param load_all_channels: Checkbutton to select all the channels of an experiment
        :param channel_layer: The channel layer of a TissueMAPS experiment
        :return: napari_layers.Image object, with access_token stored as metadata
        """
        id_exp = TissueMAPSConnectionWidget.search_id(self, experiment_name)
        if load_all_channels:
            multi_layer = []
            for chan_dict in self.channel_layer_data:
                pyramid = tissuemaps_interface(self.tm_connector.url.value,
                                               token,
                                               id_exp, str(chan_dict["id"]))
                res = (pyramid, {'name': "CL_"+str(chan_dict["id"]),
                                 'metadata': {'token': self.tm_connector.token}}, 'image')
                multi_layer.append(res)
            return multi_layer
        pyramid = tissuemaps_interface(self.tm_connector.url.value,
                                       token,
                                       id_exp, channel_layer['id'])
        return [(pyramid, {'name': "CL_"+str(channel_layer['id']),
                           'metadata': {'token': self.tm_connector.token}}, 'image')]


@napari_hook_implementation
def napari_get_reader(path):
    """
    Napari plugin that returns a reader interface for TissueMAPs .

    .. note::
       This hook does not support a list of paths

    :param path:  The path of the image
    :return: The tissuemaps_interface function or None
    """
    if isinstance(path, str) and path.endswith(".xmld"):
        return tissuemaps_connector
    return None


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    """
    Napari plugin that returns a Magicui widget
    :return: The TissueMAPS connection widget
    """
    return TissueMAPSConnectionWidget
