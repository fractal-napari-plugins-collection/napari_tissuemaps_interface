"""
Unittests for napri_tissuemaps_interface.napari_tissuemaps_interface module.
"""
import numpy as np
from types import SimpleNamespace
from PIL import Image
import io

from napari_tissuemaps_interface.napari_tissuemaps_interface import tissuemaps_connector 

EXPERIMENT_ID = 95
CHANNEL_LAYER_ID = 1
TILE_SIZE = 256
IMAGE_HEIGHT = 4096
IMAGE_WIDTH = 6144
MAX_ZOOM = 4

def test_tissuemaps_connector(mocker):
    """
    Test for the napari_tissuemaps_interface.tissuemaps_connector() function.
    """
    class MockXMLDoc():
        def __init__(self, data_dict):
            self.data_dict = data_dict
        def getElementsByTagName(self, key):
            return self.data_dict[key]

    class MockXMLElement():
        def __init__(self, attr_dict):
            self.attributes = attr_dict

    class MockXMLAttribut():
        def __init__(self, value):
            self.value = value

    mocker.patch(
        'xml.dom.minidom.parse',
        return_value=MockXMLDoc({
            'url': [
                MockXMLElement({'url': MockXMLAttribut('testurl')}),
            ],
            'user' : [MockXMLElement({
                'name': MockXMLAttribut('testusername'), 
                'password' : MockXMLAttribut('testpassword')
                })
            ],
            'layerdata' : [ MockXMLElement({
                        'experiment_id': MockXMLAttribut(EXPERIMENT_ID),
                        'channel_layer_id': MockXMLAttribut(CHANNEL_LAYER_ID)
                })
            ]
        })
    )


    def mocked_requests_post(*args, **kwargs):
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code

            def json(self):
                return self.json_data


            def raise_for_status(self):
                return None

        if 'auth' in args[0] :
            return MockResponse({"access_token": "token"}, 200)
        return MockResponse(None, 404)

    mocker.patch(
        'napari_tissuemaps_interface.napari_tissuemaps_interface.requests.post', mocked_requests_post)


    def mocked_requests_get(*args, **kwargs):
        class MockResponse:
            def __init__(self, json_data, content, status_code):
                self.json_data = json_data
                self.content = content
                self.status_code = status_code

            def json(self):
                return self.json_data

            def raise_for_status(self):
                return None

        def mock_jpegdata(x, y, max_x, max_y):
            if  x >= max_x or x < 0 or y >= max_y or y < 0:
                data = np.zeros((TILE_SIZE, TILE_SIZE))
            else:
                data = np.ones((TILE_SIZE, TILE_SIZE))
            imagedata = Image.fromarray(np.uint8(data))
            jpegdata = io.BytesIO()
            imagedata.save(jpegdata, format="jpeg")
            return jpegdata.getvalue()

        if 'channel_layers' in args[0] and 'tiles' not in args[0]:
            return MockResponse({"data": [
                {
                    'id' : CHANNEL_LAYER_ID,
                    'image_size' : {
                                    'height' : IMAGE_HEIGHT, 
                                    'width' : IMAGE_WIDTH
                                },
                    'max_zoom' : MAX_ZOOM   

                }
                ]}, None, 200)
        if 'channel_layers' in args[0] and 'tiles' in args[0]:
            x_pos = kwargs['params']['x']
            y_pos = kwargs['params']['y']
            zoom = kwargs['params']['z'] 
            x_tiles = 0
            y_tiles = 0
            image_height = IMAGE_HEIGHT
            image_width = IMAGE_WIDTH
            for i in range(MAX_ZOOM - zoom):
                image_height //= 2
                image_width //= 2
            
            max_x = image_width // TILE_SIZE
            max_y = image_height // TILE_SIZE 

            content = mock_jpegdata(x_pos, y_pos, max_x, max_y)
            return MockResponse(None, content, 200) 
        return MockResponse(None, None, 404)

    mocker.patch(
        'napari_tissuemaps_interface.napari_tissuemaps_interface.requests.get', mocked_requests_get)




    layer_data = tissuemaps_connector(...)
    print(layer_data)
    assert len(layer_data) == 1
    pyramid, _ = layer_data[0]
    assert len(pyramid) == 4 
    lvl0_array = pyramid[0]
    assert lvl0_array.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    #assert lvl0_array.sum().compute() == IMAGE_HEIGHT * IMAGE_WIDTH
