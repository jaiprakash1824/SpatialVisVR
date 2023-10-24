import os
import tifffile as tiff
import xml.etree.ElementTree as ET

class TiffImageCacheSearch:
    _instance = None
    _cache = {}
    _cache_channel_names = {}
    _image_Path = "reg002_X01_Y01_Z10"
    _basePath = r"/Volumes/Jai_Lab_T7/Data/Codex/CRC_TMA_A_hyperstacks_bestFocus/bestFocus"
    _channel_Names = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TiffImageCacheSearch, cls).__new__(cls)
        return cls._instance
    
    def set_imagePath(self, path):
        k = os.path.join(self._basePath,path)
        self._image_Path = k
    
    def get_imagePath(self):
        return self._image_Path

    def load_tiff_image(self):
        tem_path= os.path.join(self._basePath,self._image_Path+".tif")
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
        return ["HOECHST1","blank","blank","blank","HOECHST2","CD44","FOXP3","CDX2","HOECHST3","CD8","p53","GATA3","HOECHST4","CD45","T-bet","beta-catenin","HOECHST5","HLA-DR","PD-L1","Ki67","HOECHST6","CD45RA","CD4","CD21","HOECHST7","MUC-1","CD30","CD2","HOECHST8","Vimentin","CD20","LAG-3","HOECHST9","Na-K-ATPase","CD5","IDO-1","HOECHST10","Cytokeratin","CD11b","CD56","HOECHST11","aSMA","BCL-2","CD25","HOECHST12","Collagen IV","CD11c","PD-1","HOCHST13","Granzyme B","EGFR","VISTA","HOECHST14","CD15","CD194","ICOS","HOECHST15","MMP9","Synaptophysin","CD71","HOECHST16","GFAP","CD7","CD3","HOECHST17","Chromogranin A","CD163","CD57","HOECHST18","empty","CD45RO","CD68","HOECHST19","empty","CD31","Podoplanin","HOECHST20","empty","CD34","CD38","HOECHST21","empty","CD138","MMP12","HOECHST22","empty","empty","empty","HOECHST23","empty","empty","DRAQ5"]


    def get_cached_images(self):
        return list(self._cache.keys())
    
    def get_cached_image_data(self, path):
        return self._cache[path]

    def clear_cache(self):
        self._cache.clear()

    def _get_channel_names(self,ome_tiff_file):
        return ["HOECHST1","blank","blank","blank","HOECHST2","CD44","FOXP3","CDX2","HOECHST3","CD8","p53","GATA3","HOECHST4","CD45","T-bet","beta-catenin","HOECHST5","HLA-DR","PD-L1","Ki67","HOECHST6","CD45RA","CD4","CD21","HOECHST7","MUC-1","CD30","CD2","HOECHST8","Vimentin","CD20","LAG-3","HOECHST9","Na-K-ATPase","CD5","IDO-1","HOECHST10","Cytokeratin","CD11b","CD56","HOECHST11","aSMA","BCL-2","CD25","HOECHST12","Collagen IV","CD11c","PD-1","HOCHST13","Granzyme B","EGFR","VISTA","HOECHST14","CD15","CD194","ICOS","HOECHST15","MMP9","Synaptophysin","CD71","HOECHST16","GFAP","CD7","CD3","HOECHST17","Chromogranin A","CD163","CD57","HOECHST18","empty","CD45RO","CD68","HOECHST19","empty","CD31","Podoplanin","HOECHST20","empty","CD34","CD38","HOECHST21","empty","CD138","MMP12","HOECHST22","empty","empty","empty","HOECHST23","empty","empty","DRAQ5"]
