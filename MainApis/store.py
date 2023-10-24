import tifffile as tiff
import xml.etree.ElementTree as ET
import os

class TiffImageCache:
    _instance = None
    _cache = {}
    _cache_channel_names = {}
    _basePath = "/Volumes/Jai_Lab_T7/Data/Codex/OME_TIFF/Hubmap_images"
    _image_Path = "/Volumes/Jai_Lab_T7/Data/Codex/OME_TIFF/Hubmap_images/HBM233.GTZN.466"
    _channel_Names = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TiffImageCache, cls).__new__(cls)
        return cls._instance
    
    def get_all_file_names(self):
        return [d for d in os.listdir(self._basePath) if os.path.isdir(os.path.join(self._basePath, d))]

    def set_imagePath(self, path):
        k =os.path.join(self._basePath,path)
        self._image_Path = k
    
    def get_imagePath(self):
        return self._image_Path

    def load_tiff_image(self):
        tem_path= os.path.join(self._image_Path,"reg001_expr.ome.tif")
        if tem_path in self._cache:
            print("in cashe returning")
            return self._cache[tem_path]
        try:
            image = tiff.imread(tem_path)
            self._cache[tem_path] = image
            print("not in cashe returning")
            return image
        except Exception as e:
            print(f"Error loading TIFF image: {str(e)}")
            return None
        
    def get_channel_names(self):
        tem_path= os.path.join(self._image_Path,"reg001_expr.ome.tif")
        if tem_path in self._cache_channel_names:
            return self._cache_channel_names[tem_path]
        try:
            self._cache_channel_names[tem_path] = self._get_channel_names(tem_path)
            return self._cache_channel_names[tem_path]
        except Exception as e:
            print(f"Error loading TIFF image channel names: {str(e)}")
            return None

    def get_cached_images(self):
        return list(self._cache.keys())
    
    def get_cached_image_data(self, path):
        return self._cache[path]

    def clear_cache(self):
        self._cache.clear()

    def _get_channel_names(self,ome_tiff_file):
        try:
            # Open the OME-TIFF file as binary and read the OME-XML metadata
            print(ome_tiff_file)
            with open(ome_tiff_file, 'rb') as file:
                contents = file.read()
                start = contents.find(b'<?xml')
                end = contents.find(b'</OME>', start) + len(b'</OME>')
                ome_xml = contents[start:end].decode('utf-8')
                
            root = ET.fromstring(ome_xml)
            channel_names = []
            print(root)
            for i in root.findall(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel") :
                channel_names.append(str(i.get('Name')))
            return channel_names

        except Exception as e:
            print(f"Error: {e}")

        return None
