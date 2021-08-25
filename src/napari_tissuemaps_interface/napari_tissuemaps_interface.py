"""
Main module containing the napari widget for reading from a TissueMAPS interface.
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

from .lazy_array import LazyArray

TILE_SIZE = 256  # hard-coded in TissueMAPS


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
    Function which reads a channel layer from the TissueMAPs API.
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
        def __init__(self, shape, dtype, tile_size, zoom):
            super().__init__(shape, dtype, tile_size)
            self.zoom = zoom

        @dask.delayed
        def read_tile(self, y_tile, x_tile):
            api_url = (
                'api/experiments/' + str(experiment_id) + '/channel_layers/' +
                str(channel_layer_id) + '/tiles'
            )
            tiles_resp = http_get(url, api_url, token, x=x_tile, y=y_tile, z=self.zoom)
            img = PIL.Image.open(BytesIO(tiles_resp.content))
            data = np.zeros((self.tile_size, self.tile_size))
            data[:img.size[1], :img.size[0]] = np.asarray(img)
            return data

    channel_layers = get_data(
        url,
        'api/experiments/' + str(experiment_id) + '/channel_layers',
        token
    )

    channel_layer = next(
        item for item in channel_layers if item["id"] == channel_layer_id
    )

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
            # directly return the highest resolution.
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

    :param path: XML file path
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

    pyramid = tissuemaps_interface(
        auth_data['url'],
        auth_data['token'],
        query_data['experiment_id'],
        query_data['channel_layer_id']
    )

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
                "channel_name": {"choices": [""]},
            },
        )

        def get_experiment_names(*args):
            # pylint: disable=W0613
            if len(self.experiments) > 0:
                return [experiment['name'] for experiment in self.experiments]
            return []

        def get_channel_names(*args):
            # pylint: disable=W0613
            channel_names = []
            if len(self.channels) > 0:
                channel_names = [channel['name'] for channel in self.channels]
            if len(channel_names) > 1:
                channel_names.insert(0, "-- All --")
            return channel_names

        @self.tm_connector.changed.connect
        def update_experiments(event):
            # pylint: disable=W0613
            if self.tm_connector.token != "":
                resp = get_data(
                    self.tm_connector.url.value,
                    "/api/experiments",
                    self.tm_connector.token
                )
                self.experiments = resp
                self.experiment_name.choices = []
                self.experiment_name.reset_choices()

        @self.experiment_name.changed.connect
        def update_channels(event):
            # pylint: disable=W0613
            exp_id = [
                exp["id"] for exp in self.experiments if exp["name"] == self.experiment_name.value
            ][0]
            resp = get_data(
                self.tm_connector.url.value,
                'api/experiments/' + str(exp_id) + '/channels',
                self.tm_connector.token
            )
            self.channels = resp
            self.channel_name.choices = []
            self.channel_name.reset_choices()

        self.experiments = []
        self.channels = []
        self.experiment_name._default_choices = get_experiment_names
        self.channel_name._default_choices = get_channel_names

        self.native.layout().addStretch()

    def __setitem__(self, key, value):
        """Prevent assignment by index."""
        raise NotImplementedError("magicgui.Container does not support item setting.")
        # pylint: disable=C0301

    def apply(self, token=("", "", ""),
              experiment_name="",
              channel_name="") -> List[LayerDataTuple]:
        # pylint: disable=W0613
        """
        Function executed when the "Load Data" button is pressed.
        It calls the tissuemaps tissuemaps_interface and returns a napari
        Image layer

        :param token: The access token for querying TissueMAPS
        :param experiment_name: The name of a TissueMAPS experiment
        :param channel_name: The channel name of a TissueMAPS experiment
        :return: napari_layers.Image object, with access_token stored as metadata
        """
        exp_id = [exp["id"] for exp in self.experiments if exp["name"] == experiment_name][0]
        if channel_name == '-- All --':
            multi_layer = []
            for channel in self.channels:
                assert len(channel["layers"]) == 1
                for layer in channel["layers"]:
                    pyramid = tissuemaps_interface(
                        self.tm_connector.url.value,
                        token,
                        exp_id,
                        str(layer["id"])
                    )
                    res = (
                        pyramid, {
                            'name': channel['name'],
                            'metadata': {'token': self.tm_connector.token},
                            'opacity': 1.0 / len(self.channels),
                            'blending': 'additive'
                        },
                        'image'
                    )
                    multi_layer.append(res)
            return list(
                reversed(
                    sorted(multi_layer, key=lambda layer_data: layer_data[1]["name"])
                )
            )

        channels = [ch for ch in self.channels if ch["name"] == channel_name]
        assert len(channels) == 1
        layer_ids = [layer["id"] for layer in channels[0]["layers"]]
        assert len(layer_ids) == 1
        pyramid = tissuemaps_interface(
            self.tm_connector.url.value,
            token,
            exp_id,
            layer_ids[0]
        )
        return [(
            pyramid,
            {
                'name': channel_name,
                'metadata': {'token': self.tm_connector.token},
                'blending': 'additive'
            },
            'image'
        )]


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
